#!/usr/bin/env python3
"""Step 5A: Clustered LLM policy-level decision agent."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class DecisionAgentV2:
    """
    Clustered, policy-level LLM decision agent.
    Makes one LLM call per anomaly pattern instead of per row.
    """

    def __init__(self, llm_client, deployment_name: str, auto_exec_threshold: float = 0.85) -> None:
        self.client = llm_client
        self.deployment = deployment_name
        self.auto_exec_threshold = auto_exec_threshold

    # ---------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ---------------------------------------------------------
    def run(
        self,
        anomalies: List[Dict[str, Any]],
        variable_labels: Dict[str, str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Returns row-level decisions with structured actions."""
        if not anomalies:
            return []

        clustered = self._cluster_anomalies(anomalies)
        decisions: List[Dict[str, Any]] = []

        for cluster_key, cluster_rows in clustered.items():
            policy = self._reason_cluster(
                cluster_key=cluster_key,
                anomalies=cluster_rows,
                variable_labels=variable_labels,
                metadata=metadata,
            )

            for a in cluster_rows:
                decisions.append(
                    {
                        "anomaly": a,
                        "action": policy["action"],
                        "confidence": float(policy["confidence"]),
                        "reasoning": policy["reasoning"],
                    }
                )

        return decisions

    # ---------------------------------------------------------
    # STEP 1 — CLUSTERING LOGIC
    # ---------------------------------------------------------
    def _cluster_anomalies(self, anomalies: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Cluster by (metric, issue_type)."""
        clusters: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for a in anomalies:
            key = (str(a.get("metric", "UNKNOWN")), str(a.get("issue_type", "UNKNOWN")))
            clusters[key].append(a)
        return clusters

    # ---------------------------------------------------------
    # STEP 2 — LLM POLICY REASONING (ONCE PER CLUSTER)
    # ---------------------------------------------------------
    def _reason_cluster(
        self,
        cluster_key: Tuple[str, str],
        anomalies: List[Dict[str, Any]],
        variable_labels: Dict[str, str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        metric, issue_type = cluster_key

        prompt = self._build_prompt(
            metric=metric,
            issue_type=issue_type,
            anomalies=anomalies,
            variable_labels=variable_labels,
            metadata=metadata,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior data quality architect. "
                            "You recommend safe, governance-compliant correction policies."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=600,
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
        allowed_actions = {"NO_ACTION", "NULLIFY", "CAP", "IMPUTE", "RECODE"}

        action = policy.get("action", {}) if isinstance(policy, dict) else {}
        action_type = str(action.get("type", "NO_ACTION")).upper()
        if action_type not in allowed_actions:
            action_type = "NO_ACTION"

        parameters = action.get("parameters", {}) if isinstance(action.get("parameters", {}), dict) else {}
        method = str(parameters.get("method", "none")).lower()
        if method not in {"median", "mode", "min", "max", "none"}:
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

    # ---------------------------------------------------------
    # PROMPT
    # ---------------------------------------------------------
    def _build_prompt(
        self,
        metric: str,
        issue_type: str,
        anomalies: List[Dict[str, Any]],
        variable_labels: Dict[str, str],
        metadata: Dict[str, Dict[str, Any]],
    ) -> str:
        example_columns = list({str(a.get("column")) for a in anomalies})[:3]
        column_descriptions = {col: variable_labels.get(col, "No description") for col in example_columns}
        rule = metadata.get(metric, {})

        return f"""
You are deciding a DATA CORRECTION POLICY for a recurring anomaly pattern.

### Anomaly Pattern
Metric: {metric}
Issue Type: {issue_type}
Number of affected rows: {len(anomalies)}

### Column Meaning
{json.dumps(column_descriptions, indent=2)}

### Validation Rule
{json.dumps(rule, indent=2)}

### Context
- Dataset is a Brand Health Tracking (BHT) survey
- Data is used for KPI reporting and modeling
- Avoid unnecessary nullification
- Prefer minimal, conservative corrections
- NEVER invent new values

### Allowed Actions (choose ONE)
- NO_ACTION
- NULLIFY
- CAP
- IMPUTE
- RECODE

### Output Format (STRICT JSON)
{{
  "action": {{
    "type": "NO_ACTION | NULLIFY | CAP | IMPUTE | RECODE",
    "parameters": {{
      "method": "median | mode | min | max | none"
    }}
  }},
  "confidence": 0.0,
  "reasoning": "short explanation"
}}

Return ONLY valid JSON. Do not add any extra text.
"""
