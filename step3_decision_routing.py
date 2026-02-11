#!/usr/bin/env python3
"""Step 3: Decision routing agent."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


class DecisionRoutingAgent:
    def __init__(self, systemic_threshold: float = 0.25) -> None:
        self.systemic_threshold = systemic_threshold

    def run(
        self,
        anomalies: List[Dict[str, Any]],
        column_health: pd.DataFrame,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        col_health_map = dict(zip(column_health.get("column", []), column_health.get("status", [])))

        llm_queue: List[Dict[str, Any]] = []
        human_queue: List[Dict[str, Any]] = []

        for a in anomalies:
            col = a.get("column")
            status = col_health_map.get(col, "ROW_LEVEL")

            if status == "SYSTEMIC":
                human_queue.append(a)
            else:
                llm_queue.append(a)

        return llm_queue, human_queue
