#!/usr/bin/env python3
"""Step 0: Missing Value Intelligence agent (simplified V1)."""

from __future__ import annotations

import json

import pandas as pd


class MissingValueIntelligenceAgent:
    """
    Simplified Version-1 Missing Value Agent

    - Uses ONLY questionnaire_logic.json
    - No statistical inference
    - No variable labels
    - Fully vectorized (fast)
    """

    def __init__(self, questionnaire_logic_path=None):
        if questionnaire_logic_path:
            with open(questionnaire_logic_path, encoding="utf-8") as f:
                self.questionnaire_logic = json.load(f)
        else:
            self.questionnaire_logic = None

    # ------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------

    def run(self, df, column_mapping=None):
        """
        Returns:
            valid_nulls: list of (row_index, column, reason)
            invalid_missing: empty list (kept for compatibility)
            uncertain_missing: list of (row_index, column)
        """
        del column_mapping  # kept for compatibility

        valid_nulls = []
        uncertain_missing = []

        if self.questionnaire_logic:
            valid_mask = self._build_valid_null_mask(df)
        else:
            valid_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        # Only evaluate actual nulls
        null_mask = df.isna()

        # Valid nulls = nulls that satisfy questionnaire logic
        valid_null_positions = null_mask & valid_mask

        # Uncertain = nulls not explained by logic
        uncertain_positions = null_mask & (~valid_mask)

        # Extract positions
        for col in df.columns:
            valid_rows = df.index[valid_null_positions[col]]
            uncertain_rows = df.index[uncertain_positions[col]]

            for idx in valid_rows:
                valid_nulls.append((idx, col, "QUESTIONNAIRE_LOGIC"))

            for idx in uncertain_rows:
                uncertain_missing.append((idx, col))

        # invalid_missing kept empty for compatibility
        invalid_missing = []

        return valid_nulls, invalid_missing, uncertain_missing

    # ------------------------------------------------------------
    # Build Valid Null Mask (Fully Vectorized)
    # ------------------------------------------------------------

    def _build_valid_null_mask(self, df):
        valid_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        for rule in self.questionnaire_logic.get("filters", []):
            condition = rule["if"]
            target_cols = rule["then_null"]

            # Build row-level condition mask
            rule_mask = pd.Series(True, index=df.index)

            for trigger_col, trigger_val in condition.items():
                if trigger_col not in df.columns:
                    rule_mask &= False
                    continue

                if isinstance(trigger_val, dict) and "in" in trigger_val:
                    rule_mask &= df[trigger_col].astype(str).isin([str(v) for v in trigger_val["in"]])
                else:
                    rule_mask &= df[trigger_col].astype(str) == str(trigger_val)

            # Apply mask to target columns
            for target_col in target_cols:
                if target_col in df.columns:
                    valid_mask.loc[rule_mask, target_col] = True

        return valid_mask


def save_step0_outputs(valid_nulls, invalid_missing, uncertain_missing, outdir="data/outputs"):
    import os

    os.makedirs(outdir, exist_ok=True)

    valid_df = pd.DataFrame(valid_nulls, columns=["row_index", "column", "reason"])
    invalid_df = pd.DataFrame(invalid_missing, columns=["row_index", "column"])
    uncertain_df = pd.DataFrame(uncertain_missing, columns=["row_index", "column"])

    valid_path = os.path.join(outdir, "step0_valid_nulls.xlsx")
    invalid_path = os.path.join(outdir, "step0_invalid_missing.xlsx")
    uncertain_path = os.path.join(outdir, "step0_uncertain_missing.xlsx")

    valid_df.to_excel(valid_path, index=False)
    invalid_df.to_excel(invalid_path, index=False)
    uncertain_df.to_excel(uncertain_path, index=False)

    return {
        "valid_nulls": valid_path,
        "invalid_missing": invalid_path,
        "uncertain_missing": uncertain_path,
    }
