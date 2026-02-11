#!/usr/bin/env python3
"""Build master_metadata.json, column_mapping.json, and questionnaire_logic.json from SPSS (.sav) survey data."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CANONICAL_HINTS = {
    "AWARENESS": [r"\baware", r"awareness", r"know"],
    "USAGE": [r"\buse\b", r"usage", r"used", r"trial", r"activity"],
    "AFFINITY": [r"affinity", r"love", r"prefer", r"consider"],
    "UNIQUENESS": [r"unique", r"distinct"],
    "DYNAMISM": [r"dynamic", r"modern", r"innovative"],
    "PROOF": [r"\bproof\b", r"\bprof\b"],
}


@dataclass
class LLMClient:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-06-01"

    @property
    def enabled(self) -> bool:
        if self.provider == "azure":
            return bool(self.api_key and self.azure_endpoint and self.azure_deployment)
        return bool(self.api_key)

    def complete_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        import requests

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            if self.provider == "azure":
                endpoint = self.azure_endpoint.rstrip("/")
                url = (
                    f"{endpoint}/openai/deployments/{self.azure_deployment}/chat/completions"
                    f"?api-version={self.azure_api_version}"
                )
                headers = {"api-key": self.api_key, "Content-Type": "application/json"}
                payload = {
                    "messages": messages,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                }
            else:
                url = f"{self.base_url.rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                }

            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None


def normalize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
    return s.upper() or "UNKNOWN"


def infer_type_from_value_labels(value_labels: Dict[Any, Any]) -> Tuple[str, Dict[str, Any]]:
    if not value_labels:
        return "unknown", {}

    codes = list(value_labels.keys())
    numeric_codes = [c for c in codes if isinstance(c, (int, float))]

    if set(numeric_codes) == {0, 1} and len(codes) == 2:
        return "binary", {"allowed": [0, 1]}

    likert_candidates = {1, 2, 3, 4, 5, 6, 7}
    if numeric_codes and set(numeric_codes).issubset(likert_candidates):
        return "likert", {"allowed": sorted(int(x) for x in numeric_codes)}

    return "categorical", {"allowed": sorted(codes, key=lambda x: str(x))}


def extract_series_stem(column: str) -> str:
    raw = str(column).upper()
    raw = re.sub(r"[^A-Z0-9#]+", "_", raw).strip("_")
    raw = re.sub(r"#\d+$", "", raw)
    raw = re.sub(r"_\d+$", "", raw)
    return raw or "UNKNOWN"


def infer_type(series) -> Tuple[str, Dict[str, Any]]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown", {}
    if getattr(non_null, "dtype", None).kind in {"i", "u", "f"}:
        return "numeric", {}
    return "text", {}


def profile_dataframe(df, meta) -> List[Dict[str, Any]]:
    rows = []
    value_labels = meta.variable_value_labels or {}
    labels = meta.column_labels or []
    for i, col in enumerate(df.columns):
        col_value_labels = value_labels.get(col, {})
        col_type, extra = infer_type_from_value_labels(col_value_labels)
        rows.append(
            {
                "column": col,
                "label": labels[i] if i < len(labels) else None,
                "stem": extract_series_stem(col),
                "type_guess": col_type,
                "value_labels": col_value_labels,
                **extra,
            }
        )
    return rows


def heuristic_metric_name(column: str, label: Optional[str], stem: Optional[str] = None) -> str:
    hay = f"{column} {label or ''}".lower()
    for canon, patterns in CANONICAL_HINTS.items():
        if any(re.search(p, hay) for p in patterns):
            return canon
    if stem:
        return normalize_name(stem)
    return normalize_name(extract_series_stem(column))


def chunked(items: List[Any], n: int) -> List[List[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def enforce_stem_consistency(profiles: List[Dict[str, Any]], mapping: Dict[str, str]) -> Dict[str, str]:
    by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for row in profiles:
        by_stem.setdefault(row.get("stem") or extract_series_stem(row["column"]), []).append(row)

    for stem, rows in by_stem.items():
        if len(rows) < 2:
            continue

        hint_votes: List[str] = []
        map_votes: List[str] = []
        for row in rows:
            col = row["column"]
            hint_votes.append(heuristic_metric_name(col, row.get("label"), stem=stem))
            map_votes.append(mapping.get(col, normalize_name(stem)))

        chosen = next((v for v in hint_votes if v != normalize_name(stem)), None)
        if not chosen:
            freq: Dict[str, int] = {}
            for v in map_votes:
                freq[v] = freq.get(v, 0) + 1
            chosen = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[0][0]

        for row in rows:
            mapping[row["column"]] = chosen

    return mapping


def map_columns_agentic(profiles: List[Dict[str, Any]], llm: LLMClient) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    system = (
        "You map survey columns to canonical metric names. "
        "Return strict JSON {'mapping': {original: canonical}}. "
        "For brand/product variants map to one canonical name like AWARENESS/USAGE/AFFINITY."
    )

    for batch in chunked(profiles, 120):
        fallback = {r["column"]: heuristic_metric_name(r["column"], r.get("label"), r.get("stem")) for r in batch}
        user = json.dumps({"columns": batch, "fallback": fallback})
        llm_out = llm.complete_json(system, user)
        if llm_out and isinstance(llm_out.get("mapping"), dict):
            parsed = {k: normalize_name(v) for k, v in llm_out["mapping"].items()}
            mapping.update({k: parsed.get(k, fallback[k]) for k in fallback})
        else:
            mapping.update(fallback)

    return enforce_stem_consistency(profiles, mapping)


def build_master_metadata(meta, mapping: Dict[str, str]) -> Dict[str, Any]:
    grouped: Dict[str, List[str]] = {}
    for col, canon in mapping.items():
        grouped.setdefault(canon, []).append(col)

    value_labels_map = meta.variable_value_labels or {}
    col_labels: List[Optional[str]] = list(meta.column_labels or [])
    col_names: List[str] = list(meta.column_names or [])
    label_lookup: Dict[str, str] = {}
    for i, col_name in enumerate(col_names):
        lbl = col_labels[i] if i < len(col_labels) else None
        if lbl:
            label_lookup[col_name] = str(lbl)

    metadata: Dict[str, Any] = {}
    for canon, cols in grouped.items():
        merged_labels: Dict[Any, Any] = {}
        for col in cols:
            merged_labels.update(value_labels_map.get(col, {}))

        col_type, extra = infer_type_from_value_labels(merged_labels)
        entry: Dict[str, Any] = {"type": col_type, **extra}
        if merged_labels:
            entry["value_labels"] = {
                str(k): str(v) for k, v in sorted(merged_labels.items(), key=lambda x: str(x[0]))
            }

        source_labels = [label_lookup.get(c) for c in cols if label_lookup.get(c)]
        if source_labels:
            entry["source_question_labels"] = sorted(set(source_labels))[:8]

        metadata[canon] = entry
    return metadata



def _join_examples(items: List[str], limit: int = 4) -> str:
    clean = [i for i in items if i]
    if not clean:
        return ""
    show = clean[:limit]
    suffix = "" if len(clean) <= limit else ", ..."
    return ", ".join(show) + suffix


def default_description(metric_name: str, entry: Dict[str, Any]) -> str:
    pretty = metric_name.replace("_", " ").strip()
    mtype = entry.get("type") or "survey"
    value_labels = entry.get("value_labels") or {}
    source_labels = entry.get("source_question_labels") or []

    if source_labels:
        first = f"This metric represents responses to question(s) such as {_join_examples([str(x) for x in source_labels], limit=2)}."
    else:
        first = f"This metric captures {pretty.lower()} in the survey dataset."

    if value_labels:
        label_examples = [f"{k}={v}" for k, v in list(value_labels.items())[:4]]
        second = f"Values are coded as {mtype} categories, for example {_join_examples(label_examples, limit=4)}."
    elif entry.get("allowed"):
        allowed_preview = _join_examples([str(x) for x in entry.get("allowed", [])], limit=6)
        second = f"Allowed response codes include {allowed_preview}."
    else:
        second = f"It is represented as a {mtype} field."

    return f"{first} {second}"


def enrich_metadata_descriptions(metadata: Dict[str, Any], llm: LLMClient) -> Dict[str, Any]:
    system = (
        "You write informative survey metadata descriptions. "
        "Return strict JSON: {'description': '...'} with 1-2 sentences. "
        "Do not repeat the key verbatim; explain what it measures and how values are coded when labels are available."
    )

    for metric_name, entry in metadata.items():
        context = {
            "metric": metric_name,
            "type": entry.get("type"),
            "allowed": entry.get("allowed"),
            "value_labels": entry.get("value_labels"),
            "source_question_labels": entry.get("source_question_labels", []),
        }

        description = None
        if llm.enabled:
            out = llm.complete_json(system, json.dumps(context))
            if out and isinstance(out.get("description"), str):
                candidate = out["description"].strip()
                if candidate and len(candidate.split()) >= 8:
                    description = candidate

        entry["description"] = description or default_description(metric_name, entry)

    return metadata


def detect_logic_rules(df, mapping: Dict[str, str], llm: LLMClient) -> Dict[str, Any]:
    rules = []
    reverse: Dict[str, List[str]] = {}
    for col, canon in mapping.items():
        reverse.setdefault(canon, []).append(col)

    controllers = [c for c in reverse.get("AWARENESS", []) if df[c].dropna().nunique() <= 2]
    followup_canons = [c for c in ["USAGE", "AFFINITY", "UNIQUENESS", "DYNAMISM"] if c in reverse]

    for ctrl in controllers:
        then_null: List[str] = []
        for canon in followup_canons:
            for col in reverse[canon]:
                ctrl_zero = df[ctrl] == 0
                if ctrl_zero.sum() == 0:
                    continue
                null_rate = float(df.loc[ctrl_zero, col].isna().mean())
                if null_rate >= 0.9:
                    then_null.append(col)
        if then_null:
            rules.append({"if": {ctrl: 0}, "then_null": sorted(set(then_null))})

    if llm.enabled and rules:
        system = (
            "You improve questionnaire skip-logic rules. Keep only strongly supported rules. "
            "Return JSON {'filters':[{'if':{col:value},'then_null':[cols...]}]}"
        )
        out = llm.complete_json(system, json.dumps({"initial_rules": rules}))
        if out and isinstance(out.get("filters"), list):
            rules = out["filters"]

    return {"filters": rules}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    provider = args.provider.lower()
    if provider == "azure":
        api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = args.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = args.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        model = args.model or deployment or "azure-deployment"
        return LLMClient(
            provider="azure",
            model=model,
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            azure_api_version=api_version,
        )

    return LLMClient(
        provider="openai",
        model=args.model or os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sav_file", help="Path to SPSS .sav file")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--provider", choices=["openai", "azure"], default="openai")
    parser.add_argument("--model", default=None, help="Model name (OpenAI) or logical label")
    parser.add_argument("--api-key", default=None, help="API key for OpenAI/Azure")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", default=None, help="Azure deployment name")
    parser.add_argument("--azure-api-version", default=None, help="Azure API version")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import pyreadstat  # type: ignore
    except Exception as exc:
        raise SystemExit("pyreadstat is required. Install with: pip install pyreadstat pandas requests") from exc

    df, meta = pyreadstat.read_sav(args.sav_file, apply_value_formats=False)
    llm = build_llm_client(args)

    profiles = profile_dataframe(df, meta)
    mapping = map_columns_agentic(profiles, llm)
    master_metadata = build_master_metadata(meta, mapping)
    master_metadata = enrich_metadata_descriptions(master_metadata, llm)
    logic = detect_logic_rules(df, mapping, llm)

    write_json(outdir / "master_metadata.json", master_metadata)
    write_json(outdir / "column_mapping.json", mapping)
    write_json(outdir / "questionnaire_logic.json", logic)

    print(f"Provider: {llm.provider} | LLM enabled: {llm.enabled}")
    print(f"Wrote {outdir / 'master_metadata.json'}")
    print(f"Wrote {outdir / 'column_mapping.json'}")
    print(f"Wrote {outdir / 'questionnaire_logic.json'}")


if __name__ == "__main__":
    main()
