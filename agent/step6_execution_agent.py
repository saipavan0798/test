#!/usr/bin/env python3
"""Step 6: Governed execution agent (single control point)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ExecutionAgentV2:
    """
    Deterministic execution agent.
    Executes ONLY structured actions produced by DecisionAgentV2.
    No keyword parsing. No free-text interpretation.
    """

    def __init__(self, auto_exec_threshold: float = 0.85) -> None:
        self.auto_exec_threshold = auto_exec_threshold

    # --------------------------------------------------
    # MAIN ENTRY POINT
    # --------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        decisions: List[Dict[str, Any]],
        missing_value_decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_clean = df.copy()
        audit_logs: List[Dict[str, Any]] = []

        if missing_value_decisions is None:
            missing_value_decisions = []

        # -----------------------------
        # Execute anomaly corrections
        # -----------------------------
        for d in decisions:
            anomaly = d.get("anomaly", {})
            action = d.get("action", {"type": "NO_ACTION", "parameters": {}})
            confidence = float(d.get("confidence", 0.0) or 0.0)

            row = anomaly.get("row_index")
            col = anomaly.get("column")
            if row is None or col is None or col not in df_clean.columns or row not in df_clean.index:
                continue

            old_val = df_clean.loc[row, col]

            if confidence >= self.auto_exec_threshold:
                new_val = self._apply_action(df_clean, row, col, action)
                status = "AUTO_EXECUTED"
            else:
                new_val = old_val
                status = "HUMAN_REVIEW"

            audit_logs.append(
                {
                    "row_number": int(row) + 1,
                    "column": col,
                    "metric": anomaly.get("metric"),
                    "old_value": old_val,
                    "new_value": new_val,
                    "issue": anomaly.get("issue_type"),
                    "action_type": action.get("type", "NO_ACTION"),
                    "confidence": confidence,
                    "status": status,
                    "source": "DECISION_AGENT_V2",
                }
            )

        # ---------------------------------
        # Execute missing value decisions
        # ---------------------------------
        for m in missing_value_decisions:
            row = m.get("row_index")
            col = m.get("column")
            action = m.get("action", {"type": "NO_ACTION", "parameters": {}})
            confidence = float(m.get("confidence", 0.0) or 0.0)

            if row is None or col is None or col not in df_clean.columns or row not in df_clean.index:
                continue

            old_val = df_clean.loc[row, col]

            if action.get("type") == "VALID_NULL":
                new_val = old_val
                status = "SKIPPED_VALID_NULL"

            elif action.get("type") == "IMPUTE" and confidence >= self.auto_exec_threshold:
                new_val = self._impute(df_clean, col, action)
                df_clean.loc[row, col] = new_val
                status = "AUTO_IMPUTED"

            else:
                new_val = old_val
                status = "HUMAN_REVIEW"

            audit_logs.append(
                {
                    "row_number": int(row) + 1,
                    "column": col,
                    "metric": None,
                    "old_value": old_val,
                    "new_value": new_val,
                    "issue": "MISSING_VALUE",
                    "action_type": action.get("type", "NO_ACTION"),
                    "confidence": confidence,
                    "status": status,
                    "source": "MISSING_VALUE_AGENT",
                }
            )

        return df_clean, pd.DataFrame(audit_logs)

    # --------------------------------------------------
    # ACTION DISPATCHER (TOOLS)
    # --------------------------------------------------
    def _apply_action(self, df: pd.DataFrame, row: Any, col: str, action: Dict[str, Any]) -> Any:
        action_type = action.get("type", "NO_ACTION")
        params = action.get("parameters", {}) if isinstance(action.get("parameters", {}), dict) else {}

        if action_type == "NO_ACTION":
            return df.loc[row, col]

        if action_type == "NULLIFY":
            df.loc[row, col] = np.nan
            return np.nan

        if action_type == "CAP":
            method = params.get("method", "median")
            numeric_series = pd.to_numeric(df[col], errors="coerce")

            if method == "min":
                val = numeric_series.min()
            elif method == "max":
                val = numeric_series.max()
            else:
                val = numeric_series.median()

            df.loc[row, col] = val
            return val

        if action_type == "IMPUTE":
            val = self._impute(df, col, action)
            df.loc[row, col] = val
            return val

        if action_type == "RECODE":
            # Conservative default: nullify unless mapping exists
            df.loc[row, col] = np.nan
            return np.nan

        # Safe fallback
        return df.loc[row, col]

    # --------------------------------------------------
    # IMPUTATION TOOL
    # --------------------------------------------------
    def _impute(self, df: pd.DataFrame, col: str, action: Dict[str, Any]) -> Any:
        series = df[col]
        method = action.get("parameters", {}).get("method", "median")

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            if method == "min":
                return numeric_series.min()
            if method == "max":
                return numeric_series.max()
            return numeric_series.median()

        mode_series = series.mode(dropna=True)
        if mode_series.empty:
            return np.nan
        return mode_series.iloc[0]
