#!/usr/bin/env python3
"""Step 4: Deterministic rule engine for safe auto-fixes."""

from __future__ import annotations

from typing import Any, Dict, Optional


class RuleEngine:
    def auto_resolve(self, anomaly: Dict[str, Any], column_health_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
        col = anomaly.get("column")
        status = column_health_map.get(col, "ROW_LEVEL")

        # Never auto-fix systemic issues
        if status == "SYSTEMIC":
            return None

        metric = anomaly.get("metric")
        issue = anomaly.get("issue_type")

        # Only safe deterministic fix
        if metric == "AGE" and issue == "OUT_OF_RANGE":
            return {"recommended_action": "CAP", "confidence": 0.99}

        # Everything else requires reasoning
        return None
