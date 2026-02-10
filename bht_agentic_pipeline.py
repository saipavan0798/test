#!/usr/bin/env python3
"""Build master_metadata.json, column_mapping.json, and questionnaire_logic.json from SPSS (.sav) survey data.

Designed for large BHT-style datasets (1000+ columns) with optional LLM assistance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_HINTS = {
    "AWARENESS": [r"\baware", r"awareness", r"know"],
    "USAGE": [r"\buse\b", r"usage", r"used", r"trial"],
    "AFFINITY": [r"affinity", r"love", r"prefer", r"consider"],
    "UNIQUENESS": [r"unique", r"distinct"],
    "DYNAMISM": [r"dynamic", r"modern", r"innovative"],
}


@dataclass
class LLMClient:
    api_key: Optional[str]
    model: str
    base_url: str = "https://api.openai.com/v1"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        import requests

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            return None


def normalize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
    return s.upper() or "UNKNOWN"


def infer_type(series) -> Tuple[str, Dict[str, Any]]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown", {}

    uniques = non_null.unique()
    unique_count = len(uniques)

    if getattr(non_null, "dtype", None).kind in {"i", "u", "f"}:
        vals = [v for v in uniques if not (isinstance(v, float) and math.isnan(v))]
        if set(vals).issubset({0, 1}) and unique_count <= 2:
            return "binary", {"allowed": [0, 1]}
        if unique_count <= 7 and set(vals).issubset({1, 2, 3, 4, 5, 6, 7}):
            return "likert", {"allowed": sorted(int(v) for v in vals)}
        return "numeric", {
            "min": float(non_null.min()),
            "max": float(non_null.max()),
        }

    sample = [str(v) for v in non_null.head(200).tolist()]
    allowed = sorted({str(v) for v in non_null.unique().tolist()})
    if unique_count <= 20:
        return "categorical", {"allowed": allowed}
    return "text", {"sample": sample[:20]}


def profile_dataframe(df, meta) -> List[Dict[str, Any]]:
    rows = []
    value_labels = meta.variable_value_labels or {}
    labels = meta.column_labels or []
    for i, col in enumerate(df.columns):
        s = df[col]
        col_type, extra = infer_type(s)
        rows.append(
            {
                "column": col,
                "label": labels[i] if i < len(labels) else None,
                "type_guess": col_type,
                "missing_rate": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
                "value_labels": value_labels.get(col, {}),
                **extra,
            }
        )
    return rows


def heuristic_metric_name(column: str, label: Optional[str]) -> str:
    hay = f"{column} {label or ''}".lower()
    for canon, patterns in CANONICAL_HINTS.items():
        if any(re.search(p, hay) for p in patterns):
            return canon
    return normalize_name(column)


def chunked(items: List[Any], n: int) -> List[List[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def map_columns_agentic(profiles: List[Dict[str, Any]], llm: LLMClient) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    system = (
        "You map survey columns to canonical metric names. "
        "Return strict JSON {'mapping': {original: canonical}}. "
        "For brand/product variants map to one canonical name like AWARENESS/USAGE/AFFINITY."
    )

    for batch in chunked(profiles, 120):
        fallback = {r["column"]: heuristic_metric_name(r["column"], r.get("label")) for r in batch}
        user = json.dumps({"columns": batch, "fallback": fallback})
        llm_out = llm.complete_json(system, user)
        if llm_out and isinstance(llm_out.get("mapping"), dict):
            parsed = {k: normalize_name(v) for k, v in llm_out["mapping"].items()}
            mapping.update({k: parsed.get(k, fallback[k]) for k in fallback})
        else:
            mapping.update(fallback)

    return mapping


def build_master_metadata(df, mapping: Dict[str, str]) -> Dict[str, Any]:
    grouped: Dict[str, List[str]] = {}
    for col, canon in mapping.items():
        grouped.setdefault(canon, []).append(col)

    metadata: Dict[str, Any] = {}
    for canon, cols in grouped.items():
        series = df[cols].stack(dropna=True)
        t, extra = infer_type(series)
        metadata[canon] = {"type": t, **extra}
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

    if llm.enabled and len(rules) > 0:
        system = (
            "You improve questionnaire skip-logic rules. Keep only strongly supported rules. "
            "Return JSON {'filters':[{'if':{col:value},'then_null':[cols...]}]}"
        )
        user = json.dumps({"initial_rules": rules})
        out = llm.complete_json(system, user)
        if out and isinstance(out.get("filters"), list):
            rules = out["filters"]

    return {"filters": rules}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sav_file", help="Path to SPSS .sav file")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import pyreadstat  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "pyreadstat is required. Install with: pip install pyreadstat pandas requests"
        ) from exc

    df, meta = pyreadstat.read_sav(args.sav_file, apply_value_formats=False)

    llm = LLMClient(api_key=args.api_key, model=args.model, base_url=args.base_url)

    profiles = profile_dataframe(df, meta)
    mapping = map_columns_agentic(profiles, llm)
    master_metadata = build_master_metadata(df, mapping)
    logic = detect_logic_rules(df, mapping, llm)

    write_json(outdir / "master_metadata.json", master_metadata)
    write_json(outdir / "column_mapping.json", mapping)
    write_json(outdir / "questionnaire_logic.json", logic)

    print(f"Wrote {outdir / 'master_metadata.json'}")
    print(f"Wrote {outdir / 'column_mapping.json'}")
    print(f"Wrote {outdir / 'questionnaire_logic.json'}")


if __name__ == "__main__":
    main()
