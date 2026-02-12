# BHT Agentic Metadata Builder

Python pipeline to generate:
- `master_metadata.json`
- `column_mapping.json`
- `questionnaire_logic.json`
- `variable_labels.json`

from an SPSS `.sav` file containing survey / brand health tracking data (even with 1000+ columns).

## What it does

1. Reads `.sav` with labels and value metadata.
2. Profiles each column (type guess, unique count, missingness, value labels).
3. Uses an agentic mapping step to map many raw columns to one canonical metric name.
4. Builds `master_metadata.json` at canonical-metric level.
5. Builds `column_mapping.json` at raw-column level.
6. Detects questionnaire skip logic and writes `questionnaire_logic.json`.
7. Creates `variable_labels.json` with standardized human-readable labels per source column.

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

- `outputs/master_metadata.json` (now includes `description` per canonical metric; generated via LLM when configured, else deterministic 1-2 sentence fallback based on question labels and value labels)
- `outputs/column_mapping.json`
- `outputs/questionnaire_logic.json`
- `outputs/variable_labels.json`

## Notes

- For very wide surveys, mapping is chunked in batches.
- If LLM config is missing or API fails, deterministic heuristics are used.

- Skip-logic inference now enforces one-to-one follow-up assignment, so a follow-up column is linked to only one controlling question.
- Skip-logic trigger values are inferred from SPSS value-label meanings (e.g., No / Don't know / Refused), not only numeric `0`.
- If high-confidence controller→follow-up evidence is not found, questionnaire logic is intentionally returned as blank (`filters: []`).
- When LLM is enabled, questionnaire refinement now receives up to 50 sampled rows plus candidate controller/follow-up evidence and value-label context to improve funnel accuracy while preserving one-to-one follow-up assignment.
- Trigger discovery now also evaluates single-code trigger candidates per controller (in addition to negative-label triggers) to capture more valid skip funnels without forcing many-to-many mappings.


## Mapping & metadata behavior (updated)

- Numbered column families are bucketed by stem to avoid fragmented mappings (example: `PROF#1..PROF#5` -> one canonical metric, `ACTIVITY#1..#7` -> one canonical metric).
- The pipeline prioritizes SPSS `value_labels` for metadata definitions.
- `master_metadata.json` now uses `value_labels` as the source for `allowed` values and labels, instead of computing min/max from observed data.
- This makes metadata deterministic and questionnaire/codebook-aligned.


- Variable labels prefer SPSS `column_labels`; if missing, a standardized fallback label is generated from column structure and canonical mapping.


## Step 0 Agent (Missing Value Intelligence)

- Added `agent/step0_missing_value_intelligence.py` with your Step 0 flow (questionnaire logic + group-scoped statistical inference).
- Notebook now runs Step 0 after pipeline generation and saves:
  - `data/outputs/step0_valid_nulls.xlsx`
  - `data/outputs/step0_invalid_missing.xlsx`
  - `data/outputs/step0_uncertain_missing.xlsx`


## Step 1 Agent (Row-Level Anomaly Detection)

- Added `agent/step1_row_level_anomaly_detection.py` with row-level checks against metadata + column mapping.
- Notebook now runs Step 1 and keeps `anomalies` in-memory for Step 2 (no file output at Step 1).


## Step 2 Agent (Column Health Classification)

- Added `agent/step2_column_health_classification.py` to classify each column as `SYSTEMIC` vs `ROW_LEVEL` from Step 1 anomalies.
- Notebook now runs Step 2 and saves: `data/outputs/column_health_report.xlsx`.


## Step 3 Agent (Decision Routing)

- Added `agent/step3_decision_routing.py` to route row anomalies into `llm_queue` and `human_queue`.
- Routing rule: anomalies from `SYSTEMIC` columns go to human review; others go to LLM queue.
- Notebook now runs Step 3 immediately after Step 2 and prints both queue sizes.


## Step 4 Agent (Rule Engine - Deterministic Auto-Fix)

- Added `agent/step4_rule_engine.py` with deterministic auto-fix policy.
- Current safe rule: if `metric == AGE` and issue is `OUT_OF_RANGE` (and column is not systemic), recommend `CAP` with high confidence.
- Notebook now runs Step 4 after routing and exports: `data/outputs/auto_decisions.xlsx`.


## Step 5A Agent (LLM Policy Decisions - Clustered)

- Added `agent/step5a_llm_policy_decision.py` (`DecisionAgentV2`) for cluster-level LLM policy reasoning.
- Clusters anomalies by `(metric, issue_type)` and makes one LLM call per cluster (not per row).
- Notebook now runs Step 5A on `remaining_for_llm` and exports: `data/outputs/llm_decisions.xlsx`.

## Step 5B Agent (Missing Value LLM - Uncertain Cases Only)

- Added `agent/step5b_missing_value_llm.py` (`MissingValueLLMAgentV2`) for uncertain missing-value cases from Step 0.
- Clusters uncertain missing cases by `column` and makes one LLM call per column pattern.
- Allowed actions are constrained to `VALID_NULL`, `IMPUTE`, or `NO_ACTION`.
- Notebook now runs Step 5B on `uncertain_missing` and exports: `data/outputs/missing_llm_decisions.xlsx`.

## Step 6 Agent (Execution - Single Control Point)

- Added `agent/step6_execution_agent.py` (`ExecutionAgentV2`) as the governed execution layer.
- Executes only structured actions from Step 4/5A decisions and Step 5B missing-value decisions.
- Auto-executes actions only when confidence is above threshold; otherwise routes to `HUMAN_REVIEW` in audit.
- Notebook now runs Step 6 and exports:
  - `data/outputs/cleaned_df.xlsx`
  - `data/outputs/audit_log.xlsx`

## Step 7A/7B/7C Agents (Baseline + Current Stats + Drift)

- Added `agent/step7_stats_agent.py` (`StatsAgent`) to compute current numeric/categorical profile stats per mapped column.
- Added `agent/step7_baseline_stats_agent.py` (`BaselineStatsAgent`) to resolve historical baseline stats from:
  1) precomputed stats file (JSON/XLSX), or
  2) baseline dataset (CSV/XLSX) when stats file is unavailable.
- Added `agent/step7_drift_agent.py` (`DriftAgent`) for drift detection:
  - numeric: scale change, mean shift, variance shift, PSI distribution shift
  - categorical: domain drift, category-distribution PSI shift
- Notebook now exports:
  - `data/outputs/drift_alerts.xlsx`
  - `data/outputs/current_stats.json`

## Step 8 Agent (Systemic Intelligence)

- Added `agent/step8_systemic_intelligence_agent.py` (`SystemicIntelligenceAgent`) for executive risk summary generation from systemic column health + drift alerts.
- Includes deterministic fallback summary when LLM call fails.
- Notebook now exports:
  - `data/outputs/systemic_executive_summary.txt`
