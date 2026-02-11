#!/usr/bin/env python3
"""Step 7A: Resolve baseline stats from stats file or baseline dataset."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd


class BaselineStatsAgent:
    """
    Resolves baseline statistics for drift detection.

    Priority:
    1. Use baseline stats file if available (JSON or XLSX)
    2. Else compute stats from baseline dataset (CSV/XLSX)
    """

    def __init__(self, stats_agent) -> None:
        self.stats_agent = stats_agent

    def resolve(
        self,
        metadata: Dict[str, Dict[str, Any]],
        column_mapping: Dict[str, str],
        baseline_stats_path: Optional[str] = None,
        baseline_data_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if baseline_stats_path and os.path.exists(baseline_stats_path):
            print("✅ Loading baseline stats from file")
            return self._load_stats_file(baseline_stats_path)

        if baseline_data_path and os.path.exists(baseline_data_path):
            print("⚠️ Baseline stats file missing. Computing from baseline data.")
            baseline_df = self._load_dataset(baseline_data_path)
            return self.stats_agent.compute_stats(
                df=baseline_df,
                metadata=metadata,
                column_mapping=column_mapping,
            )

        raise ValueError("Neither baseline stats file nor baseline data file was found.")

    def _load_stats_file(self, path: str) -> Dict[str, Dict[str, Any]]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._load_stats_from_excel(path)

    def _load_dataset(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path)

    def _load_stats_from_excel(self, path: str) -> Dict[str, Dict[str, Any]]:
        xls = pd.ExcelFile(path)
        stats: Dict[str, Dict[str, Any]] = {}

        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            if df.empty:
                continue
            if "metric" in df.columns and "value" in df.columns:
                # optional two-column layout
                stats[sheet] = dict(zip(df["metric"], df["value"]))
                continue

            if "column" in df.columns:
                key_col = "column"
            elif "metric" in df.columns:
                key_col = "metric"
            else:
                # fallback: first column is entity key
                key_col = df.columns[0]

            rows = df.to_dict(orient="records")
            for row in rows:
                key = str(row.get(key_col))
                if key and key != "nan":
                    row.pop(key_col, None)
                    stats[key] = row

        return stats
