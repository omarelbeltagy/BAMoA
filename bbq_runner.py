# BBQ bias evaluation runner for BA-MoA
import asyncio
import json
import os
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
import random
import argparse
from collections import defaultdict
from moa_core import run_moa, parse_two_channel

def ex_key(example):
    """BBQ example_id restarts per category, so it is NOT unique after
    concatenate_datasets() — 42k collisions across the full corpus.
    Key on (category, example_id)."""
    return (example["category"], example.get("example_id"))


def result_key(result):
    m = result["bbq_metadata"]
    return (m["category"], m["example_id"])


def stratified_sample(dataset, n_per_cell=6, seed=42, exclude_ids=None):
    """
    Sample roughly evenly across (category, context_condition, question_polarity)
    cells, instead of filtering to one condition and taking the first N per
    category. Uses random sampling within each cell (not the first N in dataset
    order) to avoid any ordering bias (e.g. templates or names sub-groups
    clustered together in the source data).

    exclude_ids: optional set of example_id values to exclude from the sampling
    pool before drawing — used to top up an existing run with n_per_cell
    *additional* questions per cell, without re-selecting anything already run.

    Returns the sampled list plus a report dict of cell -> actual count taken,
    so shortfalls (a cell with fewer than n_per_cell available) are visible.
    """

    rng = random.Random(seed)
    exclude_ids = exclude_ids or set()

    cells = defaultdict(list)
    for ex in dataset:
        if ex_key(ex) in exclude_ids:
            continue
        key = (ex["category"], ex["context_condition"], ex["question_polarity"])
        cells[key].append(ex)

    subset = []
    report = {}
    for key in sorted(cells):
        pool = cells[key]
        take = min(n_per_cell, len(pool))
        if take < n_per_cell:
            print(f"  WARNING: cell {key} has only {len(pool)} examples (< {n_per_cell} requested)")
        sampled = rng.sample(pool, take)
        subset.extend(sampled)
        report[key] = take

    rng.shuffle(subset)  # avoid cell-grouped ordering in the output file
    return subset, report

def format_bbq_prompt(example):
    return (
        f"{example['context']}\n\n"
        f"Question: {example['question']}\n\n"
        f"A) {example['ans0']}\n"
        f"B) {example['ans1']}\n"
        f"C) {example['ans2']}\n\n"
        "Answer with only the letter A, B, or C."
    )


def parse_answer(text):
    """Extract A, B, or C from the beginning of a model response."""
    if not text:
        return None
    # Two-channel format first (D4); fall back to bare letter for run files
    # collected before that change.
    _, letter = parse_two_channel(text)
    if letter:
        return letter
    cleaned = str(text).strip().upper().rstrip(".,):;")
    return cleaned if cleaned in ("A", "B", "C") else None


def load_bbq():
    """Load all BBQ configs from HuggingFace parquet files, discovering configs dynamically."""
    from huggingface_hub import list_repo_files

    parquet_files = list(list_repo_files(
        "heegyu/bbq", repo_type="dataset", revision="refs/convert/parquet"
    ))
    configs = sorted(set(f.split("/")[0] for f in parquet_files if f.endswith(".parquet")))
    print(f"Found {len(configs)} BBQ categories: {configs}")

    splits = []
    for config in configs:
        ds = load_dataset(
            "parquet",
            data_files={"test": f"hf://datasets/heegyu/bbq@refs%2Fconvert%2Fparquet/{config}/test/0000.parquet"},
            split="test",
        )
        splits.append(ds)
    return concatenate_datasets(splits)


async def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue", dest="continue_path", metavar="PATH", default=None,
        help="Path to an existing outputs/bbq/run_*.json file to resume from. "
             "Already-completed questions (matched by example_id) are skipped "
             "and new results are appended to the same file.",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=20,
        help="Target number of questions per (category, context, polarity) cell "
             "for the current invocation. With --continue, this tops up each cell up to "
             "this total (existing + new), not n_per_cell additional on top.",
    )
    args = parser.parse_args(argv)

    completed_ids = set()
    all_results = []
    out_path = None
    completed_per_cell = defaultdict(int)

    if args.continue_path:
        with open(args.continue_path) as f:
            all_results = json.load(f)
        completed_ids = {result_key(r) for r in all_results}
        for r in all_results:
            m = r["bbq_metadata"]
            key = (m["category"], m["context_condition"], m["question_polarity"])
            completed_per_cell[key] += 1
        out_path = args.continue_path
        print(f"Resuming from {out_path}: {len(completed_ids)} questions already completed.")
    print("Loading BBQ dataset...")
    dataset = load_bbq()
    print(f"Loaded {len(dataset)} total examples")
    print("Fields:", list(dataset.features.keys()))
    print("\nSample example fields (non-text):")
    sample = dataset[0]
    print(json.dumps({k: sample[k] for k in ['category', 'question_polarity', 'context_condition', 'label', 'ans0', 'ans1', 'ans2']}, indent=2))

    # Stratified sample across category × context_condition × question_polarity.
    # 11 categories × 2 context conditions × 2 polarities = 44 cells.
    # e.g n_per_cell=6 → 264 total

    N_PER_CELL = args.n_per_cell

    if args.continue_path:
        # Top up each cell individually to N_PER_CELL total.
        cells_needed = defaultdict(list)
        for ex in dataset:
            if ex_key(ex) in completed_ids:
                continue
            key = (ex["category"], ex["context_condition"], ex["question_polarity"])
            cells_needed[key].append(ex)

        rng = random.Random(42)
        subset = []
        for key in sorted(cells_needed):
            need = max(0, N_PER_CELL - completed_per_cell.get(key, 0))
            pool = cells_needed[key]
            take = min(need, len(pool))
            subset.extend(rng.sample(pool, take))
        rng.shuffle(subset)
        total = len(all_results) + len(subset)
        print(f"Topping up to {N_PER_CELL}/cell: {len(subset)} new questions needed "
              f"(on top of {len(completed_ids)} already done) → target total {total}")
    else:
        subset, cell_report = stratified_sample(dataset, n_per_cell=N_PER_CELL,
                                                exclude_ids=completed_ids)
        total = len(subset)
        n_cells = len(cell_report)
        n_full_cells = sum(1 for v in cell_report.values() if v == N_PER_CELL)
        print(f"\n{n_cells} cells (category × context × polarity), "
              f"{n_full_cells}/{n_cells} fully sampled at {N_PER_CELL} each")
        print(f"Total questions: {total}")

    if out_path is None:
        os.makedirs("outputs/bbq", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/bbq/run_{timestamp}.json"

    remaining = [ex for ex in subset if ex_key(ex) not in completed_ids]
    skipped = total - len(remaining)
    if skipped:
        print(f"Skipping {skipped} already-completed questions, {len(remaining)} remaining.")

    idx_to_letter = {0: 'A', 1: 'B', 2: 'C'}

    for i, example in enumerate(remaining):
        print(f"\n[{len(completed_ids) + i + 1}/{total}] {example['category']}")

        prompt = format_bbq_prompt(example)
        run_log = await run_moa(prompt)

        correct_letter = idx_to_letter[example['label']]
        run_log["bbq_metadata"] = {
            "example_id": example.get("example_id", i),
            "category": example["category"],
            "context_condition": example["context_condition"],
            "question_polarity": example["question_polarity"],
            "label": example["label"],
            "correct_answer": correct_letter,
            "ans0": example["ans0"],
            "ans1": example["ans1"],
            "ans2": example["ans2"],
            "answer_info": example["answer_info"],
            "additional_metadata": example.get("additional_metadata"),
        }

        for layer_name, layer_responses in run_log["layers"].items():
            answers = {m.split("/")[-1]: parse_answer(r) for m, r in layer_responses.items()}
            print(f"  {layer_name}: {answers}")
        print(f"  final:   {parse_answer(run_log['final_response'])}")

        all_results.append(run_log)

        # Checkpoint after every question — safe if run is interrupted
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. {total} questions saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
