# BHT Agentic Metadata Builder

Python pipeline to generate:
- `master_metadata.json`
- `column_mapping.json`
- `questionnaire_logic.json`

from an SPSS `.sav` file containing survey / brand health tracking data (even with 1000+ columns).

## What it does

1. Reads `.sav` with labels and value metadata.
2. Profiles each column (type guess, unique count, missingness, value labels).
3. Uses an **agentic mapping step** to map many raw columns to one canonical metric name (for example many brand columns -> `AWARENESS`, `USAGE`, `AFFINITY`).
4. Builds `master_metadata.json` at the canonical-metric level.
5. Builds `column_mapping.json` at raw-column level.
6. Detects questionnaire skip logic (`if awareness == 0, then downstream brand questions are null`) and writes `questionnaire_logic.json`.

The script works in two modes:
- **LLM-assisted** (recommended): set `OPENAI_API_KEY`.
- **Heuristic fallback**: if no API key is provided, it still produces outputs using deterministic rules.

## Install

```bash
pip install pyreadstat pandas requests
```

## Run

```bash
python bht_agentic_pipeline.py /path/to/data.sav --outdir outputs
```

Optional LLM settings:

```bash
export OPENAI_API_KEY="..."
export LLM_MODEL="gpt-4.1-mini"
# optional for OpenAI-compatible providers
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Outputs

- `outputs/master_metadata.json`
- `outputs/column_mapping.json`
- `outputs/questionnaire_logic.json`

## Notes

- For very wide surveys, mapping is chunked in batches to keep token usage stable.
- Canonical names are normalized to uppercase snake case.
- You can extend canonical hint patterns in `CANONICAL_HINTS` for domain-specific metrics.
