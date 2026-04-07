# TrustBench – Institutional Trust Measurement Pipeline

Pipeline for measuring LLM institutional trust using Hydra config management and OpenRouter API.

## Project Structure

```
trustbench/
├── configs/
│   ├── config.yaml              # Main config (paths, API, countries, run settings)
│   ├── models/
│   │   └── default_models.yaml  # Models to benchmark
│   └── experiment/
│       ├── full_run.yaml        # Full experiment config
│       └── pilot.yaml           # Quick pilot config
├── data/
│   └── institutional_trust_data.csv   # WVS items (Part 1 + Q292)
├── src/
│   ├── prompt_generators/
│   │   ├── __init__.py          # Part 1: WVS confidence & politician trust
│   │   └── part2.py             # Part 2: Stated (Likert) + Revealed (delegation)
│   ├── runners/
│   │   └── __init__.py          # Async OpenRouter API runner
│   └── analysis/
│       └── __init__.py          # Scoring, reverse-coding, aggregation
├── run.py                       # Hydra entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

```bash
# Full run (all models, all countries, 5 repetitions)
python run.py

# Pilot run (1 model, 2 countries, 2 reps)
python run.py experiment=pilot

# Dry run — generates prompts without calling APIs
python run.py run.dry_run=true

# Override repetitions
python run.py run.n_repetitions=10

# Override temperature
python run.py run.temperature=0.7

# Run only Part 1
python run.py run.parts=[1]

# Run only Part 2
python run.py run.parts=[2]
```

## What Gets Generated

### Part 1 – WVS-Style (36 prompts, no country variation)
- **Q64–Q89**: Confidence in 26 organizations (4-point scale)
- **Q292**: 11 politician/government trust items (5-point scale)

### Part 2 – Stated Trust (12 countries × 8 institutions × 12 items = 1,152 prompts)
- 4 sub-scales: Overall Trust (OT), Benevolent Intent (BI), Competence (CO), Integrity/Honesty (IH)
- 3 items per sub-scale (1 reverse-coded)
- 7-point Likert scale

### Part 2 – Revealed Trust (12 countries × 48 scenarios = 576 prompts)
- **Government**: 8 delegation scenarios
- **Military**: 8 scenarios
- **Media**: 8 scenarios
- **Judiciary**: 6 scenarios
- **Elections**: 6 scenarios
- **Central Bank**: 6 scenarios
- **Police**: 6 scenarios

### Total per model per repetition: ~1,764 prompts
### With 4 models × 5 reps: ~35,280 API calls

## Output Structure

```
outputs/
├── prompts/
│   └── prompts_YYYYMMDD_HHMMSS.jsonl     # All generated prompts
└── results/
    ├── raw_results_YYYYMMDD_HHMMSS.jsonl  # Raw API responses
    └── analysis_YYYYMMDD_HHMMSS/
        ├── scored_responses.csv            # All responses with scores
        ├── part1_summary.csv               # Part 1 aggregated
        ├── stated_summary.csv              # Stated trust aggregated
        └── revealed_summary.csv            # Revealed trust P(trust) by scenario
```

## Adding Models

Edit `configs/models/default_models.yaml`:

```yaml
model_list:
  - id: "openai/gpt-4o"
    label: "GPT-4o"
  - id: "your-new/model-id"
    label: "Your Model Label"
```

## Adding Countries

Edit `configs/config.yaml` under the `countries` section. Countries are grouped by regime type for analysis convenience.

## Adding Scenarios

Edit `src/prompt_generators/part2.py` — add new tuples to `REVEALED_SCENARIOS`. Each entry is `(scenario_id, institution_key, template_text)` where `{country}` is substituted at generation time.
