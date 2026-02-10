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

## 1) How to use your actual `.sav` file

You do **not** upload the file into the script itself. You place the file on disk and pass its path.

### Option A: File is already on your machine/server

Example:

```bash
python bht_agentic_pipeline.py "/absolute/path/to/your_survey_file.sav" --outdir outputs
```

or from current folder:

```bash
python bht_agentic_pipeline.py ./data/your_survey_file.sav --outdir outputs
```

### Option B: You need to copy file from local laptop to remote Linux server

```bash
scp /local/path/your_survey_file.sav user@server:/remote/project/data/
```

Then SSH and run:

```bash
python bht_agentic_pipeline.py /remote/project/data/your_survey_file.sav --outdir outputs
```

### Option C: Notebook/Colab-style upload

Upload the file using the notebook UI, note uploaded path, then run:

```bash
python bht_agentic_pipeline.py /content/your_survey_file.sav --outdir outputs
```

## 2) API key and API setup

The script accepts API settings in **two ways**:

- Environment variables (recommended)
- CLI arguments (`--api-key`, `--model`, `--base-url`)

### A. Environment variables (recommended)

```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4.1-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python bht_agentic_pipeline.py ./data/your_survey_file.sav --outdir outputs
```

### B. CLI arguments

```bash
python bht_agentic_pipeline.py ./data/your_survey_file.sav \
  --outdir outputs \
  --api-key "sk-..." \
  --model "gpt-4.1-mini" \
  --base-url "https://api.openai.com/v1"
```

### OpenAI-compatible providers

If you use an OpenAI-compatible provider, keep the same script and change:

- `OPENAI_BASE_URL` to that provider endpoint
- `OPENAI_API_KEY` to that provider key
- `LLM_MODEL` to a model name that provider supports

## 3) Quick run checklist

1. Install dependencies.
2. Place `.sav` file in a known path.
3. Set API vars (optional, for LLM mode).
4. Run script.
5. Check generated files in output directory.

## Outputs

- `outputs/master_metadata.json`
- `outputs/column_mapping.json`
- `outputs/questionnaire_logic.json`

## Notes

- For very wide surveys, mapping is chunked in batches to keep token usage stable.
- Canonical names are normalized to uppercase snake case.
- You can extend canonical hint patterns in `CANONICAL_HINTS` for domain-specific metrics.
- If API key is missing/invalid, mapping still runs via deterministic heuristics.
