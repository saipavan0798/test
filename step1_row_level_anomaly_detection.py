#!/usr/bin/env python3
"""Step 1: Row-level anomaly detection agent."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class DetectionAgent:
    def __init__(self) -> None:
        pass

    def run(self, df: pd.DataFrame, metadata: Dict[str, Any], column_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        anomalies: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            for col in df.columns:
                if col not in column_mapping:
                    continue

                metric = column_mapping[col]
                if metric not in metadata:
                    continue

                rule = metadata[metric]
                val = row[col]
                rtype = rule.get("type")

                if rtype == "numeric":
                    self._check_numeric(anomalies, idx, col, metric, val, rule)
                elif rtype == "binary":
                    self._check_binary(anomalies, idx, col, metric, val, rule)
                elif rtype == "likert":
                    self._check_likert(anomalies, idx, col, metric, val, rule)
                elif rtype == "categorical":
                    self._check_categorical(anomalies, idx, col, metric, val, rule)

        return anomalies

    def _normalize_allowed(self, allowed):
        if allowed is None:
            return set()
        norm = set()
        for x in allowed:
            norm.add(x)
            norm.add(str(x))
            try:
                norm.add(float(x))
            except Exception:
                pass
        return norm

    def _check_numeric(self, anomalies, idx, col, metric, val, rule):
        if pd.isna(val):
            return

        min_v = rule.get("min")
        max_v = rule.get("max")

        if min_v is None or max_v is None:
            # fallback to allowed if min/max absent in value-label-driven metadata
            allowed = self._normalize_allowed(rule.get("allowed", []))
            if allowed and val not in allowed and str(val) not in allowed:
                anomalies.append(
                    self._make_record(
                        idx,
                        col,
                        metric,
                        val,
                        "INVALID_NUMERIC_CODE",
                        f"Allowed {sorted(str(x) for x in rule.get('allowed', []))}",
                    )
                )
            return

        try:
            if val < min_v or val > max_v:
                anomalies.append(
                    self._make_record(
                        idx,
                        col,
                        metric,
                        val,
                        "OUT_OF_RANGE",
                        f"Expected [{min_v}, {max_v}]",
                    )
                )
        except Exception:
            anomalies.append(
                self._make_record(
                    idx,
                    col,
                    metric,
                    val,
                    "NUMERIC_TYPE_MISMATCH",
                    f"Expected numeric within [{min_v}, {max_v}]",
                )
            )

    def _check_binary(self, anomalies, idx, col, metric, val, rule):
        if pd.isna(val):
            return

        allowed = self._normalize_allowed(rule.get("allowed", []))
        if allowed and (val not in allowed and str(val) not in allowed):
            anomalies.append(
                self._make_record(
                    idx,
                    col,
                    metric,
                    val,
                    "INVALID_BINARY",
                    f"Allowed {rule.get('allowed', [])}",
                )
            )

    def _check_likert(self, anomalies, idx, col, metric, val, rule):
        if pd.isna(val):
            return

        allowed = self._normalize_allowed(rule.get("allowed", []))
        if allowed and (val not in allowed and str(val) not in allowed):
            anomalies.append(
                self._make_record(
                    idx,
                    col,
                    metric,
                    val,
                    "INVALID_LIKERT",
                    f"Allowed scale {rule.get('allowed', [])}",
                )
            )

    def _check_categorical(self, anomalies, idx, col, metric, val, rule):
        if pd.isna(val):
            return

        allowed = self._normalize_allowed(rule.get("allowed", []))
        if allowed and (val not in allowed and str(val) not in allowed):
            anomalies.append(
                self._make_record(
                    idx,
                    col,
                    metric,
                    val,
                    "INVALID_CATEGORY",
                    f"Allowed {rule.get('allowed', [])}",
                )
            )

    def _make_record(self, row, col, metric, value, issue, rule):
        row_num = int(row) + 1 if isinstance(row, (int, np.integer)) else row
        return {
            "row_index": row,
            "row_number": row_num,
            "column": col,
            "metric": metric,
            "value": value,
            "issue_type": issue,
            "rule": rule,
        }

