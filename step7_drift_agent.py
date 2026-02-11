#!/usr/bin/env python3
"""Step 7C: Drift detection between current and baseline stats."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class DriftAgent:
    def __init__(self, mean_z_thresh: float = 3.0, std_pct_thresh: float = 0.5, psi_thresh: float = 0.2) -> None:
        self.mean_z_thresh = mean_z_thresh
        self.std_pct_thresh = std_pct_thresh
        self.psi_thresh = psi_thresh

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

    def _numeric_drift(self, col: str, cur: Dict[str, Any], hist: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        # Scale drift
        if cur.get("max") > hist.get("max") or cur.get("min") < hist.get("min"):
            alerts.append(
                {
                    "column": col,
                    "drift_type": "SCALE_CHANGE",
                    "historical_range": [hist.get("min"), hist.get("max")],
                    "current_range": [cur.get("min"), cur.get("max")],
                }
            )

        # Mean drift
        hist_std = float(hist.get("std", 0.0) or 0.0)
        if hist_std > 0:
            z = abs(float(cur.get("mean", 0.0)) - float(hist.get("mean", 0.0))) / hist_std
            if z > self.mean_z_thresh:
                alerts.append(
                    {
                        "column": col,
                        "drift_type": "MEAN_SHIFT",
                        "z_score": round(float(z), 2),
                        "historical_mean": hist.get("mean"),
                        "current_mean": cur.get("mean"),
                    }
                )

        # Std drift
        if hist_std > 0:
            std_change = abs(float(cur.get("std", 0.0)) - hist_std) / hist_std
            if std_change > self.std_pct_thresh:
                alerts.append(
                    {
                        "column": col,
                        "drift_type": "VARIANCE_SHIFT",
                        "historical_std": hist.get("std"),
                        "current_std": cur.get("std"),
                    }
                )

        # PSI drift
        psi = self._psi(hist.get("psi_bins", []), cur.get("psi_bins", []))
        if psi > self.psi_thresh:
            alerts.append(
                {
                    "column": col,
                    "drift_type": "DISTRIBUTION_SHIFT",
                    "psi": round(float(psi), 3),
                }
            )

        return alerts

    def _categorical_drift(self, col: str, cur: Dict[str, Any], hist: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        hist_dist = hist.get("distribution", {})
        cur_dist = cur.get("distribution", {})

        hist_keys = set(hist_dist.keys())
        cur_keys = set(cur_dist.keys())

        # Domain drift
        new_vals = list(cur_keys - hist_keys)
        if new_vals:
            alerts.append(
                {
                    "column": col,
                    "drift_type": "DOMAIN_DRIFT",
                    "new_categories": new_vals,
                }
            )

        # PSI drift
        psi = self._psi_cat(hist.get("psi_bins", {}), cur.get("psi_bins", {}))
        if psi > self.psi_thresh:
            alerts.append(
                {
                    "column": col,
                    "drift_type": "CATEGORY_DISTRIBUTION_SHIFT",
                    "psi": round(float(psi), 3),
                }
            )

        return alerts

    def _psi(self, expected: List[float], actual: List[float]) -> float:
        psi = 0.0
        for e, a in zip(expected, actual):
            if e == 0 or a == 0:
                continue
            psi += (a - e) * np.log(a / e)
        return float(psi)

    def _psi_cat(self, expected: Dict[str, float], actual: Dict[str, float]) -> float:
        psi = 0.0
        all_keys = set(expected.keys()).union(actual.keys())

        for k in all_keys:
            e = float(expected.get(k, 0.0001))
            a = float(actual.get(k, 0.0001))
            psi += (a - e) * np.log(a / e)

        return float(psi)
