#!/usr/bin/env python3
"""Step 2: Column health classification agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class ColumnHealthAgent:
    def __init__(self, threshold: float = 0.25) -> None:
        self.threshold = threshold

    def run(self, anomalies: List[Dict[str, Any]], total_rows: int) -> pd.DataFrame:
        # Handle zero-anomaly case safely
        if not anomalies or total_rows <= 0:
            return pd.DataFrame(columns=["column", "anomaly_count", "anomaly_pct", "status"])

        df = pd.DataFrame(anomalies)
        if "column" not in df.columns:
            return pd.DataFrame(columns=["column", "anomaly_count", "anomaly_pct", "status"])

        col_stats = df.groupby("column").size().reset_index(name="anomaly_count")
        col_stats["anomaly_pct"] = col_stats["anomaly_count"] / float(total_rows)
        col_stats["status"] = col_stats["anomaly_pct"].apply(
            lambda x: "SYSTEMIC" if x >= self.threshold else "ROW_LEVEL"
        )
        return col_stats.sort_values(["status", "anomaly_pct"], ascending=[True, False]).reset_index(drop=True)


def save_step2_output(column_health: pd.DataFrame, outdir: str = "data/outputs") -> str:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "column_health_report.xlsx"
    column_health.to_excel(report_path, index=False)
    return str(report_path)
