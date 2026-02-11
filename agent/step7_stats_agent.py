#!/usr/bin/env python3
"""Step 7B: Compute current stats for drift detection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class StatsAgent:
    def __init__(self, bins: int = 10) -> None:
        self.bins = bins

    def compute_stats(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Dict[str, Any]],
        column_mapping: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}

        for col, metric in column_mapping.items():
            if col not in df.columns:
                continue

            rule = metadata.get(metric, {})
            dtype = rule.get("type")

            series = df[col].dropna()
            if series.empty:
                continue

            if dtype in ["numeric", "likert", "binary"]:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                if numeric_series.empty:
                    continue
                stats[col] = self._numeric_stats(numeric_series)

            elif dtype == "categorical":
                stats[col] = self._categorical_stats(series)

        return stats

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        hist, bin_edges = np.histogram(series, bins=self.bins)
        hist_sum = hist.sum()
        hist_pct = (hist / hist_sum) if hist_sum > 0 else np.zeros_like(hist, dtype=float)

        std = float(series.std())
        if np.isnan(std):
            std = 0.0

        return {
            "type": "numeric",
            "mean": float(series.mean()),
            "std": std,
            "min": float(series.min()),
            "max": float(series.max()),
            "quantiles": [
                float(series.quantile(0.25)),
                float(series.quantile(0.50)),
                float(series.quantile(0.75)),
            ],
            "psi_bins": hist_pct.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        dist = series.value_counts(normalize=True).to_dict()

        # keep keys JSON-stable for downstream serialization
        dist_str = {str(k): float(v) for k, v in dist.items()}

        return {
            "type": "categorical",
            "distribution": dist_str,
            "psi_bins": dist_str,
        }
