#!/usr/bin/env python3
"""Step 7C: Drift detection between current and baseline stats (robust version)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class DriftAgent:
    def __init__(
        self,
        mean_z_thresh: float = 3.0,
        std_pct_thresh: float = 0.5,
        psi_thresh: float = 0.2,
    ) -> None:
        self.mean_z_thresh = mean_z_thresh
        self.std_pct_thresh = std_pct_thresh
        self.psi_thresh = psi_thresh

    # --------------------------------------------------
    # PUBLIC ENTRY
    # --------------------------------------------------
    def detect(
        self,
        current_stats: Dict[str, Dict[str, Any]],
        historical_stats: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        for col, cur in current_stats.items():
            if col not in historical_stats:
                continue

            hist = historical_stats[col]

            if cur.get("type") == "numeric" and hist.get("type") == "numeric":
                alerts.extend(self._numeric_drift(col, cur, hist))

            elif cur.get("type") == "categorical" and hist.get("type") == "categorical":
                alerts.extend(self._categorical_drift(col, cur, hist))

        return alerts

    # --------------------------------------------------
    # NUMERIC DRIFT
    # --------------------------------------------------
    def _numeric_drift(self, col: str, cur: Dict[str, Any], hist: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        # Safe extraction
        hist_min = hist.get("min")
        hist_max = hist.get("max")
        cur_min = cur.get("min")
        cur_max = cur.get("max")

        hist_mean = float(hist.get("mean", 0.0) or 0.0)
        cur_mean = float(cur.get("mean", 0.0) or 0.0)

        hist_std = float(hist.get("std", 0.0) or 0.0)
        cur_std = float(cur.get("std", 0.0) or 0.0)

        # ---------------- Scale Drift ----------------
        if hist_min is not None and hist_max is not None and cur_min is not None and cur_max is not None:
            if cur_max > hist_max or cur_min < hist_min:
                alerts.append(
                    {
                        "column": col,
                        "drift_type": "SCALE_CHANGE",
                        "historical_range": [hist_min, hist_max],
                        "current_range": [cur_min, cur_max],
                    }
                )

        # ---------------- Mean Drift ----------------
        if hist_std > 0:
            z = abs(cur_mean - hist_mean) / hist_std
            if z > self.mean_z_thresh:
                alerts.append(
                    {
                        "column": col,
                        "drift_type": "MEAN_SHIFT",
                        "z_score": round(float(z), 3),
                        "historical_mean": hist_mean,
                        "current_mean": cur_mean,
                    }
                )

        # ---------------- Variance Drift ----------------
        # Avoid division by zero
        if hist_std > 0:
            std_change = abs(cur_std - hist_std) / hist_std
            if std_change > self.std_pct_thresh:
                alerts.append(
                    {
                        "column": col,
                        "drift_type": "VARIANCE_SHIFT",
                        "historical_std": hist_std,
                        "current_std": cur_std,
                        "pct_change": round(float(std_change), 3),
                    }
                )

        # ---------------- PSI Drift ----------------
        psi = self._psi(
            hist.get("psi_bins", []),
            cur.get("psi_bins", []),
        )

        if psi > self.psi_thresh:
            alerts.append(
                {
                    "column": col,
                    "drift_type": "DISTRIBUTION_SHIFT",
                    "psi": round(float(psi), 4),
                }
            )

        return alerts

    # --------------------------------------------------
    # CATEGORICAL DRIFT
    # --------------------------------------------------
    def _categorical_drift(self, col: str, cur: Dict[str, Any], hist: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        hist_dist = hist.get("distribution", {}) or {}
        cur_dist = cur.get("distribution", {}) or {}

        hist_keys = set(hist_dist.keys())
        cur_keys = set(cur_dist.keys())

        # -------- Domain Drift --------
        new_vals = list(cur_keys - hist_keys)
        if new_vals:
            alerts.append(
                {
                    "column": col,
                    "drift_type": "DOMAIN_DRIFT",
                    "new_categories": new_vals,
                }
            )

        # -------- PSI Drift --------
        psi = self._psi_cat(
            hist.get("psi_bins", {}) or {},
            cur.get("psi_bins", {}) or {},
        )

        if psi > self.psi_thresh:
            alerts.append(
                {
                    "column": col,
                    "drift_type": "CATEGORY_DISTRIBUTION_SHIFT",
                    "psi": round(float(psi), 4),
                }
            )

        return alerts

    # --------------------------------------------------
    # PSI FOR NUMERIC
    # --------------------------------------------------
    def _psi(self, expected: List[float], actual: List[float]) -> float:
        if not expected or not actual:
            return 0.0

        psi = 0.0

        for e, a in zip(expected, actual):
            e = float(e)
            a = float(a)

            # Avoid log(0) or division by zero
            if e <= 0 or a <= 0:
                continue

            psi += (a - e) * np.log(a / e)

        return float(psi)

    # --------------------------------------------------
    # PSI FOR CATEGORICAL
    # --------------------------------------------------
    def _psi_cat(self, expected: Dict[str, float], actual: Dict[str, float]) -> float:
        psi = 0.0
        epsilon = 1e-6

        all_keys = set(expected.keys()).union(actual.keys())

        for k in all_keys:
            e = float(expected.get(k, epsilon))
            a = float(actual.get(k, epsilon))

            if e <= 0:
                e = epsilon
            if a <= 0:
                a = epsilon

            psi += (a - e) * np.log(a / e)

        return float(psi)


# Alias requested by notebook snippet
DriftAgent1 = DriftAgent
