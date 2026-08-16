# BA-MoA — Usage Guide

Technical documentation only

---

## Setup

```bash
export TOGETHER_API_KEY=<your_key>
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required for winobias_runner.py
```

---

## Files

| File | Purpose |
|---|---|
| `moa_core.py` | Shared MoA pipeline (models, layers, prompts). Imported by all runners — not run directly. |
| `bbq_runner.py` | Runs the BBQ benchmark through the MoA pipeline, saves raw output. |
| `bbq_scorer.py` | Scores a BBQ output file, prints metrics. |
| `winobias_runner.py` | Runs the WinoBias benchmark through the MoA pipeline, saves raw output. |
| `winobias_scorer.py` | Scores a WinoBias output file, prints metrics. |
| `app.py` | Unified entry point — auto-detects and continues the latest run per dataset. |

---

## Running via `app.py` (recommended)

```bash
# Run (or auto-continue) an experiment
python app.py --dataset bbq
python app.py --dataset winobias
```

`app.py` looks for the most recent file matching `outputs/<dataset>/run_*.json`.
If one exists, it resumes from it (tops up sampling to the target count).
If none exists, it starts a fresh run.

### Flags

| Flag | Applies to | Effect |
|---|---|---|
| `--dataset {bbq,winobias}` | required | Which experiment to run. |
| `--fresh` | both | Start a fresh run, not continuing on previously saved output |
| `--n-per-cell N` | both | Override the runner's default sample size per cell. |
| `--include-type-2` | winobias only | Also run the fixed 100-question Type-2 sanity check. |

---

## Running a runner directly (bypassing `app.py`)

Each runner also works standalone and accepts its own flags:

```bash
# BBQ
python bbq_runner.py                                  # fresh run, default n-per-cell
python bbq_runner.py --n-per-cell 40                   # fresh run, custom size
python bbq_runner.py --continue outputs/bbq/run_<ts>.json           # top up existing file
python bbq_runner.py --continue outputs/bbq/run_<ts>.json --n-per-cell 40   # top up to a new target

# WinoBias
python winobias_runner.py
python winobias_runner.py --n-per-cell 100
python winobias_runner.py --include-type-2
python winobias_runner.py --continue outputs/winobias/run_<ts>.json
python winobias_runner.py --continue outputs/winobias/run_<ts>.json --include-type-2
```

`--continue` semantics: tops up each sampling cell **to** the target count
— it does not add the target count again on top of what's already there.

---

## Where output is saved