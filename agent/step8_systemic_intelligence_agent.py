#!/usr/bin/env python3
"""Step 8: Systemic intelligence executive summary agent."""

from __future__ import annotations

from typing import Any, Dict, List


class SystemicIntelligenceAgent:
    def __init__(self, llm_client, deployment_name: str) -> None:
        self.client = llm_client
        self.deployment = deployment_name

    def generate_recommendations(self, column_health_report, drift_alerts: List[Dict[str, Any]]) -> str:
        systemic_columns: List[str] = []
        if column_health_report is not None and not column_health_report.empty:
            systemic_columns = column_health_report[column_health_report["status"] == "SYSTEMIC"]["column"].tolist()

        drift_columns: List[str] = []
        if drift_alerts:
            drift_columns = list({str(d.get("column")) for d in drift_alerts if d.get("column") is not None})

        if not systemic_columns and not drift_columns:
            return (
                "No systemic data quality or drift risks detected. "
                "Dataset is stable and suitable for analytics consumption."
            )

        prompt = self._build_prompt(systemic_columns, drift_columns)

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Chief Data Officer writing an executive risk assessment.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=700,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return self._fallback_summary(systemic_columns, drift_columns)

    def _build_prompt(self, systemic_columns: List[str], drift_columns: List[str]) -> str:
        systemic_text = ", ".join(systemic_columns) if systemic_columns else "None detected"
        drift_text = ", ".join(drift_columns) if drift_columns else "None detected"

        return f"""
You are a Chief Data Officer.

Below are enterprise data quality risk signals:

Systemic anomaly columns:
{systemic_text}

Drift detected in columns:
{drift_text}

Write a concise executive summary covering:

1. Root cause hypothesis
2. Business risk assessment
3. Technical remediation steps
4. Governance recommendations

Use bullet points.
Keep professional tone.
Limit to 10 bullet points.
"""

    def _fallback_summary(self, systemic_columns: List[str], drift_columns: List[str]) -> str:
        lines = ["- Executive summary generated via fallback (LLM unavailable)."]
        lines.append(f"- Systemic anomaly columns: {', '.join(systemic_columns) if systemic_columns else 'None detected'}.")
        lines.append(f"- Drift columns: {', '.join(drift_columns) if drift_columns else 'None detected'}.")
        lines.append("- Root cause hypothesis: schema/process changes or upstream data capture inconsistency.")
        lines.append("- Business risk: KPI comparability and model stability may be impacted.")
        lines.append("- Technical remediation: validate mappings, recheck rules, and backfill baseline where needed.")
        lines.append("- Governance: enforce drift monitoring alerts and weekly data quality review.")
        return "\n".join(lines)
