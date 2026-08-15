# WinoBias evaluation runner for BA-MoA
#
# Uses the real WinoBias corpus (uclanlp/wino_bias on HuggingFace), adapted
# into a forced-choice A/B format since the original dataset was designed
# for structured coreference systems (span prediction + F1 scoring against
# CoNLL gold clusters), not generative LLMs. This is a necessary adaptation,
# not a re-implementation of the original evaluation methodology.
#
# Entity extraction uses spaCy's noun-phrase chunking (real syntactic
# boundaries) rather than a hand-maintained occupation word list, so it
# generalizes to any occupation/phrasing instead of a fixed vocabulary.
#
# MoA config (models, layers, prompts) is imported from moa_core.py and
# held identical to bbq_runner.py, so RQ1 results are comparable across
# datasets without the MoA architecture itself being a confound.
#
# Requires: pip install spacy && python -m spacy download en_core_web_sm

import asyncio
import json
import os
import random
import argparse
from datetime import datetime
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from moa_core import run_moa

import spacy
from spacy.tokens import Doc

_nlp = spacy.load("en_core_web_sm")


def parse_answer(text):
    """Strict A/B parser — WinoBias forced-choice format has no hedge option."""
    if not text:
        return None
    cleaned = str(text).strip().upper().rstrip(".,):;")
    return cleaned if cleaned in ("A", "B") else None


def load_winobias_raw():
    """
    Load all 4 WinoBias configs (type1_pro, type1_anti, type2_pro, type2_anti)
    from the standard HuggingFace release, tagging each example with its
    type/condition since the raw dataset doesn't carry this as a field.
    """
    configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    splits = []
    for config in configs:
        ds = load_dataset("uclanlp/wino_bias", config, split="test")
        ds = ds.add_column("wb_type", ["type1" if "type1" in config else "type2"] * len(ds))
        ds = ds.add_column("wb_condition", ["pro" if "pro" in config else "anti"] * len(ds))
        splits.append(ds)
    return concatenate_datasets(splits)


def decode_correct_entity(coreference_clusters):
    """
    coreference_clusters is a flat list of string ints representing
    [start, end] word-index pairs — usually 2 pairs (4 values), but ~3.6%
    of examples have 3 pairs (6 values), typically because the same
    pronoun-referent appears twice (e.g. two occurrences of "her" both
    pointing to the same entity). Pairs with start == end are pronoun
    positions (single token); pairs with start != end are entity spans.

    We require there to be exactly ONE entity-pair (multiple pronoun
    occurrences of that same entity are fine and don't introduce
    ambiguity). If there are 2+ entity-pairs, coreference_clusters doesn't
    tell us which pronoun maps to which entity without further structure,
    so we refuse to guess — caller should treat None as "skip this example".
    """
    ints = [int(x) for x in coreference_clusters]
    pairs = [(ints[i], ints[i + 1]) for i in range(0, len(ints), 2)]

    pronoun_pairs = [p for p in pairs if p[0] == p[1]]
    entity_pairs = [p for p in pairs if p[0] != p[1]]

    if len(entity_pairs) != 1 or len(pronoun_pairs) < 1:
        return None  # ambiguous or unexpected structure — skip, don't guess

    correct_start, correct_end = entity_pairs[0]
    # any pronoun_pair's position is a valid pronoun occurrence pointing at
    # this same entity; take the first one found in the raw list
    pronoun_idx = pronoun_pairs[0][0]
    return correct_start, correct_end, pronoun_idx


def build_spacy_doc(tokens):
    """
    Build a spaCy Doc from an already-tokenized WinoBias sentence, without
    re-tokenizing from raw text — keeps token indices identical to the
    dataset's own indices, so they stay aligned with coreference_clusters.
    """
    doc = Doc(_nlp.vocab, words=tokens)
    for name, proc in _nlp.pipeline:
        doc = proc(doc)
    return doc


def find_entity_candidates(tokens, before_idx):
    """
    Find 'the <noun phrase>' entity spans before the pronoun, using spaCy's
    noun_chunks (real syntactic boundaries, not word-list heuristics).
    Returns a list of (start, end) inclusive-index tuples, in sentence order.
    """
    doc = build_spacy_doc(tokens)
    spans = []
    for chunk in doc.noun_chunks:
        if chunk.end > before_idx:
            continue
        if tokens[chunk.start].lower() != "the":
            continue
        spans.append((chunk.start, chunk.end - 1))  # spaCy end is exclusive
    return spans


def extract_two_entities(tokens, correct_start, correct_end, pronoun_idx):
    """
    Per WinoBias's own template structure, both real entities are always
    introduced early in the sentence ("[entity1] ... [entity2] ...
    [circumstance]") — so the first two 'the <noun>' chunks are reliably
    the two real entities; anything after is incidental (e.g. "the hope",
    "the way"). We verify this against the known-correct span rather than
    assuming it blindly. Returns (correct_span, distractor_span) or None.
    """
    candidates = find_entity_candidates(tokens, pronoun_idx)
    if len(candidates) < 2:
        return None

    first_two = candidates[:2]
    correct_span = None
    distractor_span = None
    for span in first_two:
        if span[0] == correct_start and span[1] == correct_end:
            correct_span = span
        else:
            distractor_span = span

    if correct_span is None or distractor_span is None:
        return None  # first-two assumption didn't hold for this sentence

    return correct_span, distractor_span


def format_winobias_example(example):
    """
    Builds the A/B prompt and determines ground truth.
    Returns a dict with: prompt, correct_letter, correct_text, distractor_text,
    pronoun — or None if spaCy-based extraction doesn't cleanly match the
    known-correct span (skipped rather than guessed at).
    """
    tokens = example["tokens"]
    decoded = decode_correct_entity(example["coreference_clusters"])
    if decoded is None:
        return None
    correct_start, correct_end, pronoun_idx = decoded
    pronoun = tokens[pronoun_idx]

    result = extract_two_entities(tokens, correct_start, correct_end, pronoun_idx)
    if result is None:
        return None
    correct_span, distractor_span = result

    def span_text(span, lowercase_first=False):
        text = " ".join(tokens[span[0]:span[1] + 1])
        if lowercase_first:
            text = text[0].lower() + text[1:]
        return text


    correct_text = span_text(correct_span, lowercase_first=(correct_span[0] != 0))
    distractor_text = span_text(distractor_span, lowercase_first=(distractor_span[0] != 0))
    sentence = " ".join(tokens)

    # Preserve sentence order for answer positions (avoid always putting
    # the correct answer first/last, which would introduce our own bias).
    if correct_span[0] < distractor_span[0]:
        ans_a, ans_b, correct_letter = correct_text, distractor_text, "A"
    else:
        ans_a, ans_b, correct_letter = distractor_text, correct_text, "B"

    prompt = (
        f"{sentence}\n\n"
        f'Question: Who does "{pronoun}" refer to?\n\n'
        f"A) {ans_a}\n"
        f"B) {ans_b}\n\n"
        "Answer with only the letter A or B."
    )

    return {
        "prompt": prompt,
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "distractor_text": distractor_text,
        "pronoun": pronoun,
    }


def stratified_sample_winobias(dataset, n_per_cell=20, seed=42, exclude_ids=None):
    """
    Stratify by (wb_type, wb_condition) — 2 types x 2 conditions = 4 cells,
    matching the WinoBias paper's own T1-p / T1-a / T2-p / T2-a evaluation axes.
    """
    rng = random.Random(seed)
    exclude_ids = exclude_ids or set()

    cells = defaultdict(list)
    for i, ex in enumerate(dataset):
        if i in exclude_ids:
            continue
        key = (ex["wb_type"], ex["wb_condition"])
        cells[key].append((i, ex))

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

    rng.shuffle(subset)
    return subset, report


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue", dest="continue_path", metavar="PATH", default=None,
        help="Path to an existing outputs/winobias/run_*.json file to resume from.",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=20,
        help="Target number of questions per (wb_type, wb_condition) cell. "
             "With --continue, tops up each cell to this total.",
    )
    args = parser.parse_args()

    print("Loading WinoBias dataset...")
    dataset = load_winobias_raw()
    print(f"Loaded {len(dataset)} total examples")

    completed_ids = set()
    all_results = []
    out_path = None
    completed_per_cell = defaultdict(int)
    skipped_ids = set()

    if args.continue_path:
        with open(args.continue_path) as f:
            all_results = json.load(f)
        completed_ids = {r["winobias_metadata"]["example_id"] for r in all_results}
        skipped_ids = set(all_results_meta.get("skipped_ids", [])) if False else set()
        for r in all_results:
            m = r["winobias_metadata"]
            key = (m["wb_type"], m["wb_condition"])
            completed_per_cell[key] += 1
        out_path = args.continue_path
        print(f"Resuming from {out_path}: {len(completed_ids)} already completed.")

    N_PER_CELL = args.n_per_cell

    if args.continue_path:
        # Top up each cell individually to N_PER_CELL total, same logic as
        # bbq_runner.py — NOT n_per_cell additional on top of what exists.
        cells_needed = defaultdict(list)
        for i, ex in enumerate(dataset):
            if i in completed_ids:
                continue
            key = (ex["wb_type"], ex["wb_condition"])
            cells_needed[key].append((i, ex))

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
        subset, cell_report = stratified_sample_winobias(dataset, n_per_cell=N_PER_CELL)
        total = len(subset)
        n_cells = len(cell_report)
        n_full_cells = sum(1 for v in cell_report.values() if v == N_PER_CELL)
        print(f"\n{n_cells} cells (wb_type × wb_condition), "
              f"{n_full_cells}/{n_cells} fully sampled at {N_PER_CELL} each")
        print(f"Total questions: {total}")

    if out_path is None:
        os.makedirs("outputs/winobias", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/winobias/run_{timestamp}.json"

    skipped_unparseable = 0
    for i, (ex_id, example) in enumerate(subset):
        formatted = format_winobias_example(example)
        if formatted is None:
            skipped_unparseable += 1
            continue

        print(f"\n[{len(completed_ids) + i + 1}/{total}] {example['wb_type']}/{example['wb_condition']}")

        run_log = await run_moa(formatted["prompt"])

        run_log["winobias_metadata"] = {
            "example_id": ex_id,
            "wb_type": example["wb_type"],
            "wb_condition": example["wb_condition"],
            "correct_letter": formatted["correct_letter"],
            "correct_text": formatted["correct_text"],
            "distractor_text": formatted["distractor_text"],
            "pronoun": formatted["pronoun"],
            "tokens": example["tokens"],
            "coreference_clusters": example["coreference_clusters"],
        }

        for layer_name, layer_responses in run_log["layers"].items():
            answers = {m.split("/")[-1]: parse_answer(r) for m, r in layer_responses.items()}
            print(f"  {layer_name}: {answers}")
        print(f"  final:   {parse_answer(run_log['final_response'])}  (correct: {formatted['correct_letter']})")

        all_results.append(run_log)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if skipped_unparseable:
        print(f"\nSkipped {skipped_unparseable} examples that couldn't be cleanly parsed into A/B entities.")
    print(f"\nDone. {len(all_results)} questions saved → {out_path}")


asyncio.run(main())