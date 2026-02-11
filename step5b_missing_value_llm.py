#!/usr/bin/env python3
"""Step 5B: Clustered LLM decisions for uncertain missing values."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List


class MissingValueLLMAgentV2:
    """
    Clustered LLM agent for resolving UNCERTAIN missing values only.
    Operates at column-pattern level, not row level.
    """

    def __init__(self, client, deployment_name: str) -> None:
        self.client = client
        self.deployment = deployment_name

    # --------------------------------------------------
    # PUBLIC ENTRY POINT
    # --------------------------------------------------
    def run(
        self,
        uncertain_cases: List[Dict[str, Any]],
        variable_labels: Dict[str, str],
        column_mapping: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        if not uncertain_cases:
            return []

        clusters = self._cluster_cases(uncertain_cases)
        decisions: List[Dict[str, Any]] = []

        for col, rows in clusters.items():
            policy = self._reason_cluster(
                column=col,
                rows=rows,
                variable_labels=variable_labels,
                column_mapping=column_mapping,
            )

            for r in rows:
                decisions.append(
                    {
                        "row_index": r.get("row_index"),
                        "column": col,
                        "action": policy["action"],
                        "confidence": float(policy["confidence"]),
                        "reasoning": policy["reasoning"],
                        "source": "MISSING_VALUE_LLM_V2",
                    }
                )

        return decisions

    # --------------------------------------------------
    # STEP 1 — CLUSTER BY COLUMN
    # --------------------------------------------------
    def _cluster_cases(self, cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in cases:
            column = c.get("column")
            if column is None:
                continue
            clusters[str(column)].append(c)
        return clusters

    # --------------------------------------------------
    # STEP 2 — LLM REASONING (ONCE PER COLUMN)
    # --------------------------------------------------
    def _reason_cluster(
        self,
        column: str,
        rows: List[Dict[str, Any]],
        variable_labels: Dict[str, str],
        column_mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        metric = column_mapping.get(column, "UNKNOWN")

        prompt = self._build_prompt(
            column=column,
            metric=metric,
            rows=rows,
            variable_labels=variable_labels,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior survey methodologist. "
                            "You decide whether missing survey values are valid skips "
                            "or true missing data. Be conservative."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=500,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            policy = json.loads(text)
        except Exception:
            policy = {
                "action": {"type": "NO_ACTION", "parameters": {}},
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM output",
            }

        return self._sanitize_policy(policy)

    def _sanitize_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        allowed_actions = {"VALID_NULL", "IMPUTE", "NO_ACTION"}

        action = policy.get("action", {}) if isinstance(policy, dict) else {}
        action_type = str(action.get("type", "NO_ACTION")).upper()
        if action_type not in allowed_actions:
            action_type = "NO_ACTION"

        parameters = action.get("parameters", {}) if isinstance(action.get("parameters", {}), dict) else {}
        method = str(parameters.get("method", "none")).lower()
        if method not in {"median", "mode", "none"}:
            method = "none"

        confidence_raw = policy.get("confidence", 0.0) if isinstance(policy, dict) else 0.0
        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(policy.get("reasoning", "No reasoning provided")) if isinstance(policy, dict) else "No reasoning provided"

        return {
            "action": {"type": action_type, "parameters": {"method": method}},
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------
    def _build_prompt(
        self,
        column: str,
        metric: str,
        rows: List[Dict[str, Any]],
        variable_labels: Dict[str, str],
    ) -> str:
        description = variable_labels.get(column, "No description")

        return f"""
You are evaluating UNCERTAIN missing values in a survey dataset.

### Column
Name: {column}
Metric: {metric}
Description: {description}

### Context
- Dataset is Brand Health Tracking (BHT)
- Missing values may be due to survey skip logic
- Avoid unnecessary imputation
- Prefer VALID_NULL if defensible
- Do NOT invent data

### Observations
- Number of rows affected: {len(rows)}

### Allowed Actions (choose ONE)
- VALID_NULL        → missing is logically correct
- IMPUTE            → missing should be filled conservatively
- NO_ACTION         → escalate for human review

### Output Format (STRICT JSON)
{{
  "action": {{
    "type": "VALID_NULL | IMPUTE | NO_ACTION",
    "parameters": {{
      "method": "median | mode | none"
    }}
  }},
  "confidence": 0.0,
  "reasoning": "short explanation"
}}

Return ONLY valid JSON. No extra text.
"""
