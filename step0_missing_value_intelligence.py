#!/usr/bin/env python3
"""Step 0: Missing Value Intelligence agent."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class MissingValueIntelligenceAgent:
    def __init__(
        self,
        questionnaire_logic_path: Optional[str] = None,
        variable_labels_path: Optional[str] = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        """
        questionnaire_logic_path : JSON file defining skip rules (optional)
        variable_labels_path     : JSON mapping column -> human-readable label (mandatory)
        confidence_threshold     : Statistical confidence cutoff
        """
        self.confidence_threshold = confidence_threshold

        self.questionnaire_logic: Optional[Dict[str, Any]] = None
        if questionnaire_logic_path:
            with open(questionnaire_logic_path, encoding="utf-8") as f:
                self.questionnaire_logic = json.load(f)

        if not variable_labels_path:
            raise ValueError("Variable labels file is mandatory for MissingValueIntelligenceAgent")

        with open(variable_labels_path, encoding="utf-8") as f:
            self.variable_labels: Dict[str, str] = json.load(f)

        self.logical_groups = self._build_logical_groups()
        self._group_index = self._build_group_index()

    # ------------------------------------------------------------
    # Step A — Build semantic logical groups from variable labels
    # ------------------------------------------------------------

    def _build_logical_groups(self) -> Dict[str, List[str]]:
        """
        Groups columns by brand/question blocks using semantic similarity.
        Example:
            BrandA_awareness → group: BrandA
        """
        groups: Dict[str, List[str]] = defaultdict(list)
        for col in self.variable_labels.keys():
            base = str(col).split("_")[0]
            groups[base].append(col)
        return dict(groups)

    def _build_group_index(self) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}
        for _, cols in self.logical_groups.items():
            for col in cols:
                index[col] = cols
        return index

    # ------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------

    def run(self, df: pd.DataFrame, column_mapping: Dict[str, str]):
        del column_mapping  # kept for interface consistency with pipeline stages

        valid_nulls: List[Tuple[Any, str, str]] = []
        invalid_missing: List[Tuple[Any, str]] = []
        uncertain_missing: List[Tuple[Any, str]] = []

        for col in df.columns:
            missing_mask = df[col].isna()
            if int(missing_mask.sum()) == 0:
                continue

            for idx in df.index[missing_mask]:
                if self._questionnaire_rule_applies(df, idx, col):
                    valid_nulls.append((idx, col, "QUESTIONNAIRE_LOGIC"))
                    continue

                stat_decision, confidence = self._statistical_inference(df, idx, col)
                if stat_decision and confidence >= self.confidence_threshold:
                    valid_nulls.append((idx, col, "STATISTICAL_PATTERN"))
                    continue

                uncertain_missing.append((idx, col))

        return valid_nulls, invalid_missing, uncertain_missing

    # ------------------------------------------------------------
    # Step B — Questionnaire Logic Engine
    # ------------------------------------------------------------

    def _questionnaire_rule_applies(self, df: pd.DataFrame, idx: Any, col: str) -> bool:
        if not self.questionnaire_logic:
            return False

        for rule in self.questionnaire_logic.get("filters", []):
            cond = rule.get("if", {})
            targets = rule.get("then_null", [])

            if col not in targets:
                continue

            satisfied = True
            for trigger_col, trigger_val in cond.items():
                observed = df.loc[idx, trigger_col] if trigger_col in df.columns else None

                if isinstance(trigger_val, dict) and "in" in trigger_val:
                    allowed = {str(v) for v in trigger_val["in"]}
                    if str(observed) not in allowed:
                        satisfied = False
                        break
                else:
                    if pd.isna(observed) and pd.isna(trigger_val):
                        continue
                    if observed != trigger_val:
                        satisfied = False
                        break

            if satisfied:
                return True

        return False

    # ------------------------------------------------------------
    # Step C — Statistical Inference Engine (Group-scoped)
    # ------------------------------------------------------------

    def _statistical_inference(self, df: pd.DataFrame, idx: Any, col: str) -> Tuple[bool, float]:
        group = self._get_logical_group(col)
        if not group:
            return False, 0.0

        predictors = [c for c in group if c != col and c in df.columns]
        if not predictors:
            return False, 0.0

        patterns: List[float] = []

        for p in predictors:
            if pd.isna(df.loc[idx, p]):
                continue

            mask = df[p].notna()
            if int(mask.sum()) < 20:
                continue

            same_value_mask = mask & (df[p] == df.loc[idx, p])
            denom = int(same_value_mask.sum())
            if denom < 10:
                continue

            null_rate = float(df.loc[same_value_mask, col].isna().mean())
            if not np.isnan(null_rate):
                patterns.append(null_rate)

        if not patterns:
            return False, 0.0

        prob = float(np.mean(patterns))
        return prob > self.confidence_threshold, prob

    # ------------------------------------------------------------
    # Helper: find logical group
    # ------------------------------------------------------------

    def _get_logical_group(self, col: str) -> Optional[List[str]]:
        return self._group_index.get(col)


def to_dataframe(records: List[Tuple[Any, ...]], columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=columns)


def save_step0_outputs(
    valid_nulls: List[Tuple[Any, str, str]],
    invalid_missing: List[Tuple[Any, str]],
    uncertain_missing: List[Tuple[Any, str]],
    outdir: str = "data/outputs",
) -> Dict[str, str]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    valid_df = to_dataframe(valid_nulls, ["row_index", "column", "reason"])
    invalid_df = to_dataframe(invalid_missing, ["row_index", "column"])
    uncertain_df = to_dataframe(uncertain_missing, ["row_index", "column"])

    valid_path = out / "step0_valid_nulls.xlsx"
    invalid_path = out / "step0_invalid_missing.xlsx"
    uncertain_path = out / "step0_uncertain_missing.xlsx"

    valid_df.to_excel(valid_path, index=False)
    invalid_df.to_excel(invalid_path, index=False)
    uncertain_df.to_excel(uncertain_path, index=False)

    return {
        "valid_nulls": str(valid_path),
        "invalid_missing": str(invalid_path),
        "uncertain_missing": str(uncertain_path),
    }
