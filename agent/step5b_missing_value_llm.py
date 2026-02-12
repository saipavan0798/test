#!/usr/bin/env python3
"""Step 5B: Missing-value LLM decisions for uncertain cases (hackathon mode)."""

from __future__ import annotations

import json
from collections import defaultdict


class MissingValueLLMAgentV2:
    """
    Hackathon-Optimized Version

    - Calls LLM per column (like original design)
    - BUT only for top N columns by volume of uncertain missing
    - Remaining columns → NO_ACTION
    """

    def __init__(self, client, deployment_name, max_llm_columns=2):
        self.client = client
        self.deployment = deployment_name
        self.max_llm_columns = max_llm_columns

    # --------------------------------------------------
    # PUBLIC ENTRY POINT
    # --------------------------------------------------
    def run(self, uncertain_cases, variable_labels, column_mapping):
        clusters = self._cluster_cases(uncertain_cases)

        # Sort columns by number of uncertain rows (descending)
        sorted_columns = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

        # Select top N columns for LLM
        llm_columns = set(sorted_columns[: self.max_llm_columns])

        decisions = []

        for col, rows in clusters.items():
            if col in llm_columns:
                policy = self._reason_cluster(col, rows, variable_labels, column_mapping)
            else:
                # Auto fallback for non-priority columns
                policy = {
                    "action": {"type": "NO_ACTION", "parameters": {"method": "none"}},
                    "confidence": 0.0,
                    "reasoning": "Not prioritized in hackathon mode",
                }

            for r in rows:
                decisions.append(
                    {
                        "row_index": r["row_index"],
                        "column": col,
                        "action": policy["action"],
                        "confidence": policy["confidence"],
                        "reasoning": policy["reasoning"],
                        "source": "MISSING_VALUE_LLM_V2",
                    }
                )

        return decisions

    # --------------------------------------------------
    # CLUSTER BY COLUMN
    # --------------------------------------------------
    def _cluster_cases(self, cases):
        clusters = defaultdict(list)
        for c in cases:
            clusters[c["column"]].append(c)
        return clusters

    # --------------------------------------------------
    # LLM CALL
    # --------------------------------------------------
    def _reason_cluster(self, column, rows, variable_labels, column_mapping):
        metric = column_mapping.get(column, "UNKNOWN")
        description = variable_labels.get(column, "No description")

        prompt = f"""
You are evaluating UNCERTAIN missing values in a survey dataset.

Column: {column}
Metric: {metric}
Description: {description}

Number of rows affected: {len(rows)}

Allowed Actions:
- VALID_NULL
- NO_ACTION

Return STRICT JSON:

{{
  "action": {{"type": "VALID_NULL | NO_ACTION", "parameters": {{"method": "none"}}}},
  "confidence": 0.0,
  "reasoning": "short explanation"
}}

Return ONLY valid JSON.
"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a senior survey methodologist. Be conservative."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=300,
        )

        text = response.choices[0].message.content.strip()

        try:
            return json.loads(text)
        except Exception:
            return {
                "action": {"type": "NO_ACTION", "parameters": {"method": "none"}},
                "confidence": 0.0,
                "reasoning": "LLM parsing failed",
            }
