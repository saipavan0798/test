# BHT Agentic Metadata Builder

Python pipeline to generate:
- `master_metadata.json`
- `column_mapping.json`
- `questionnaire_logic.json`

from an SPSS `.sav` file containing survey / brand health tracking data (even with 1000+ columns).

## What it does

1. Reads `.sav` with labels and value metadata.
2. Profiles each column (type guess, unique count, missingness, value labels).
3. Uses an agentic mapping step to map many raw columns to one canonical metric name.
4. Builds `master_metadata.json` at canonical-metric level.
5. Builds `column_mapping.json` at raw-column level.
6. Detects questionnaire skip logic and writes `questionnaire_logic.json`.

## Install (Anaconda recommended)

```bash
conda create -n bht-agentic python=3.11 -y
conda activate bht-agentic
pip install pyreadstat pandas requests python-dotenv jupyter
```

## Use your `.sav` file

Place the file in your project, for example:

```bash
./data/your_survey_file.sav
```

Run:

```bash
python bht_agentic_pipeline.py ./data/your_survey_file.sav --outdir outputs
```

## API setup

### OpenAI mode (default)

```bash
export OPENAI_API_KEY="..."
export LLM_MODEL="gpt-4.1-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"

python bht_agentic_pipeline.py ./data/your_survey_file.sav --outdir outputs --provider openai
```

### Azure OpenAI mode

Create `.env` (or `.evn`, both are supported in notebook):

```env
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_API_VERSION=2024-06-01
```

Then either export them in shell or keep them in `.env` and run from notebook.

CLI example:

```bash
python bht_agentic_pipeline.py ./data/your_survey_file.sav \
  --outdir outputs \
  --provider azure \
  --azure-endpoint "https://your-resource-name.openai.azure.com" \
  --azure-deployment "your-chat-deployment-name" \
  --azure-api-version "2024-06-01" \
  --api-key "your_azure_openai_key"
```

## Notebook usage

A ready notebook is included:

- `bht_agentic_pipeline_azure.ipynb`

It shows:
- loading config from `.env`/`.evn`
- reading `.sav` with `pyreadstat`
- running `bht_agentic_pipeline.py` with `--provider azure`
- verifying generated JSON outputs

Launch:

```bash
jupyter notebook bht_agentic_pipeline_azure.ipynb
```

## Outputs

- `outputs/master_metadata.json`
- `outputs/column_mapping.json`
- `outputs/questionnaire_logic.json`

## Notes

- For very wide surveys, mapping is chunked in batches.
- If LLM config is missing or API fails, deterministic heuristics are used.


## Mapping & metadata behavior (updated)

- Numbered column families are bucketed by stem to avoid fragmented mappings (example: `PROF#1..PROF#5` -> one canonical metric, `ACTIVITY#1..#7` -> one canonical metric).
- The pipeline prioritizes SPSS `value_labels` for metadata definitions.
- `master_metadata.json` now uses `value_labels` as the source for `allowed` values and labels, instead of computing min/max from observed data.
- This makes metadata deterministic and questionnaire/codebook-aligned.
