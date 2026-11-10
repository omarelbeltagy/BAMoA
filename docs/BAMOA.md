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
| `bbq_groups.py` | Resolves BBQ options to target / non-target groups. Imported by `bbq_scorer.py`. |
| `winobias_runner.py` | Runs the WinoBias benchmark through the MoA pipeline, saves raw output. |
| `winobias_scorer.py` | Scores a WinoBias output file, prints metrics. |
| `report_generator.py` | Builds a combined Markdown report (BBQ + WinoBias) with CIs. |
| `test_bbq_scorer_synthetic.py` | Synthetic-model validation for the BBQ scorer. |
| `test_winobias_scorer_synthetic.py` | Synthetic-model validation for the WinoBias scorer. |
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
| `--pool-variant V` | both | Proposer prompt variant applied to **all** proposers. `neutral` (default) is the k=0 baseline; `strong` applies the calibrated bias-injection persona (k=4). Variants are defined in `PROPOSER_PROMPTS` in `moa_core.py`. 
| `--include-type-2` | winobias only | Also run the fixed 100-question Type-2 sanity check. |

---

## Running a runner directly (bypassing `app.py`)

Each runner also works standalone and accepts its own flags:

```bash
# BBQ
python bbq_runner.py                                  # fresh run, default n-per-cell
python bbq_runner.py --n-per-cell 40                   # fresh run, custom size
python bbq_runner.py --n-per-cell 40 --pool-variant strong   # bias-injected arm
python bbq_runner.py --continue outputs/bbq/run_<ts>.json           # top up existing file
python bbq_runner.py --continue outputs/bbq/run_<ts>.json --n-per-cell 40   # top up to a new target

# WinoBias
python winobias_runner.py
python winobias_runner.py --n-per-cell 100
python winobias_runner.py --n-per-cell 196 --pool-variant strong
python winobias_runner.py --include-type-2
python winobias_runner.py --continue outputs/winobias/run_<ts>.json
python winobias_runner.py --continue outputs/winobias/run_<ts>.json --include-type-2
```

`--continue` semantics: tops up each sampling cell **to** the target count
— it does not add the target count again on top of what's already there.

`--pool-variant` note: arms are distinguished only by the run file's
`config.pool` field, not by the filename. Both arms draw the same items
(sampling seed is fixed), so comparisons across arms are paired. Do not
`--continue` a file collected under a different variant — the runner will
not detect the mismatch and the file would contain mixed conditions.

---

- Filenames are timestamped at the start of a fresh run.
- `--continue` writes to the **same file** it resumed from (no new file
  created).
- Files are checkpointed after every question — safe to `Ctrl+C` at any
  point without losing completed data.
- `app.py --fresh` (or a runner with no `--continue`) always creates a new
  timestamped file rather than overwriting an existing one.

### Output file structure

Each file is a JSON list of question records:

```json
{
  "question": "...",
  "config": {
    "pool": [["<model>", "<variant>"], ...],
    "n_layers": 4,
    "aggregator": "<model>",
    "agg_prompt_variant": "neutral",
    "proposer_synth_variant": "neutral",
    "two_channel": true,
    "seed": 1409311875,
    "temperature": 0.0
  },
  "layers": {
    "layer_1": {"<model>": "<response>", ...},
    "layer_2": {...},
    "layer_3": {...}
  },
  "layer_meta": {
    "layer_1": {"<model>": {"finish_reason": "stop", "reasoning": "...",
                            "null_reason": null, "attempts": 1,
                            "latency_s": 1.95}, ...}
  },
  "dropped_peers": {"layer_1": 0, "layer_2": 0, "layer_3": 0},
  "peer_order": {"layer_2": [2,1,3,0], "layer_3": [...], "final": [...]},
  "final_meta": { ... },
  "final_response": "...",
  "bbq_metadata": { ... }        // bbq_runner.py only
  "winobias_metadata": { ... }   // winobias_runner.py only
}
```
Proposer responses use the two-channel format:

```
REASON: <one or two sentences>
ANSWER: <the letter only>
```

Scorers parse the `ANSWER` field; the full block propagates to later layers.
`peer_order` records the permutation in which peers were shown at each
layer; `bbq_metadata.option_order` records the answer-option permutation,
which the scorer inverts.
---

## Scoring

```bash
python bbq_scorer.py outputs/bbq/run_<timestamp>.json
python winobias_scorer.py outputs/winobias/run_<timestamp>.json
```

Both scorers can be run on a **partial/in-progress** output file (safe to
run while a runner is still going in another terminal, since files are
only ever appended to, not truncated) — copy the file first if you want to
avoid any risk of reading mid-write:

```bash
cp outputs/bbq/run_<timestamp>.json /tmp/check.json
python bbq_scorer.py /tmp/check.json
```

---

## Typical workflow

```bash
# 1. Start a run
python app.py --dataset winobias

# 2. Stop anytime (Ctrl+C) — progress is saved

# 3. Resume later — automatically continues the same file
python app.py --dataset winobias

# 4. Check results at any point
python winobias_scorer.py outputs/winobias/run_<timestamp>.json
```
---

## Reporting

```bash
python report_generator.py                       # latest run per dataset
python report_generator.py --winobias outputs/winobias/run_<ts>.json
```

Writes `reports/report_<timestamp>.md` with bootstrap CIs, paired
layer-difference intervals, and detectability baselines. Bootstrapping is
the slow part — expect a few minutes.

---

## Validation

```bash
python test_bbq_scorer_synthetic.py outputs/bbq/run_<ts>.json
python test_winobias_scorer_synthetic.py outputs/winobias/run_<ts>.json
```

Pushes synthetic models (always-correct, always-stereotype, random)
through the scorers and asserts the scores they must produce. Run after
any change to scoring code. Requires enough items to clear `MIN_VALID_N`
in every bucket.

---