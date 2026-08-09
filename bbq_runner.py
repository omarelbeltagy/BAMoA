# BBQ bias evaluation runner for BA-MoA
import asyncio
import json
import os
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from together import AsyncTogether, Together
import random
from collections import defaultdict

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

reference_models = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-20b",
    "google/gemma-4-31B-it",
    "deepseek-ai/DeepSeek-V4-Pro",
]
aggregator_model = "nvidia/nemotron-3-ultra-550b-a55b"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
layers = 3

def stratified_sample(dataset, n_per_cell=6, seed=42):
    """
    Sample roughly evenly across (category, context_condition, question_polarity)
    cells, instead of filtering to one condition and taking the first N per
    category. Uses random sampling within each cell (not the first N in dataset
    order) to avoid any ordering bias (e.g. templates or names sub-groups
    clustered together in the source data).

    Returns the sampled list plus a report dict of cell -> actual count taken,
    so shortfalls (a cell with fewer than n_per_cell available) are visible.
    """
    rng = random.Random(seed)

    cells = defaultdict(list)
    for ex in dataset:
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

def get_system_prompt_with_references(prev_responses):
    return (
        aggregator_system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(r)}" for i, r in enumerate(prev_responses)])
    )


async def run_llm(model, user_prompt, prev_response=None):
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {"role": "system", "content": get_system_prompt_with_references(prev_response)},
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            if not content:
                await asyncio.sleep(sleep_time)
                continue
            return content
        except Exception as e:
            print(f"  Error [{model.split('/')[-1]}]: {e}")
            await asyncio.sleep(sleep_time)
    return None


async def run_moa(user_prompt):
    """Run the full MoA pipeline on a single prompt and return per-layer responses."""
    run_log = {"question": user_prompt, "layers": {}}

    results = await asyncio.gather(*[run_llm(model, user_prompt) for model in reference_models])
    run_log["layers"]["layer_1"] = dict(zip(reference_models, results))

    for layer_idx in range(1, layers - 1):
        results = await asyncio.gather(
            *[run_llm(model, user_prompt, prev_response=results) for model in reference_models]
        )
        run_log["layers"][f"layer_{layer_idx + 1}"] = dict(zip(reference_models, results))

    final_response = ""
    final_stream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": get_system_prompt_with_references(results)},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    for chunk in final_stream:
        if chunk.choices:
            final_response += chunk.choices[0].delta.content or ""

    run_log["final_response"] = final_response
    return run_log


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
    for char in text.strip().upper()[:20]:
        if char in ('A', 'B', 'C'):
            return char
    return None


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


async def main():
    print("Loading BBQ dataset...")
    dataset = load_bbq()
    print(f"Loaded {len(dataset)} total examples")
    print("Fields:", list(dataset.features.keys()))
    print("\nSample example fields (non-text):")
    sample = dataset[0]
    print(json.dumps({k: sample[k] for k in ['category', 'question_polarity', 'context_condition', 'label', 'ans0', 'ans1', 'ans2']}, indent=2))

    # Stratified sample across category × context_condition × question_polarity.
    # 11 categories × 2 context conditions × 2 polarities = 44 cells.
    # n_per_cell=6 → 264 total (within the 200-300 target range).
    N_PER_CELL = 6
    subset, cell_report = stratified_sample(dataset, n_per_cell=N_PER_CELL)
    total = len(subset)
    n_cells = len(cell_report)
    n_full_cells = sum(1 for v in cell_report.values() if v == N_PER_CELL)
    print(f"\n{n_cells} cells (category × context × polarity), "
          f"{n_full_cells}/{n_cells} fully sampled at {N_PER_CELL} each")
    print(f"Total questions: {total}")

    os.makedirs("outputs/bbq", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"outputs/bbq/run_{timestamp}.json"

    idx_to_letter = {0: 'A', 1: 'B', 2: 'C'}
    all_results = []

    for i, example in enumerate(subset):
        print(f"\n[{i+1}/{total}] {example['category']}")

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


asyncio.run(main())
