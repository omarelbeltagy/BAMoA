# test_winobias_scorer_synthetic.py — synthetic-model validation for the WinoBias
# scorer. A scorer that passes these is measuring stereotype alignment.
#
# WinoBias differs structurally from BBQ: the pro/anti split is a property
# of the ITEM (which corpus config it came from), not something derived
# from answer metadata. So the always-stereotype model here is one that is
# correct on pro items and wrong on anti items — mirroring the definition
# of Gap = acc_pro − acc_anti.
import json, sys, random
from winobias_scorer import score_responses, compute_diff, MIN_VALID_N


def _other(letter):
    return "B" if letter == "A" else "A"


def fake_responses(questions, mode, seed=0):
    """mode: 'correct' | 'stereo' | 'random' | 'always_a'"""
    rng = random.Random(seed)
    out = []
    for q in questions:
        m = q["winobias_metadata"]
        correct = m["correct_letter"]
        if mode == "correct":
            ans = correct
        elif mode == "stereo":
            # Right when the stereotype points at the correct answer (pro),
            # wrong when it points away (anti).
            ans = correct if m["wb_condition"] == "pro" else _other(correct)
        elif mode == "always_a":
            ans = "A"
        else:
            ans = rng.choice(["A", "B"])
        out.append({"fake": ans})
    return out


def main(path):
    with open(path) as f:
        qs = [q for q in json.load(f)
              if q["winobias_metadata"]["wb_type"] == "type1"]
    print(f"{len(qs)} Type-1 questions\n")

    n_pro = sum(1 for q in qs if q["winobias_metadata"]["wb_condition"] == "pro")
    print(f"pro={n_pro}  anti={len(qs) - n_pro}\n")

    for mode, expect in [("correct", "~0"), ("stereo", "+100%"),
                         ("random", "~0"), ("always_a", "~0")]:
        overall, _ = score_responses(fake_responses(qs, mode), qs)
        gap, acc_p, n_p, acc_a, n_a = compute_diff(overall)
        g = f"{gap:+.1%}" if gap is not None else "N/A"
        print(f"{mode:<9} Gap={g:>8}  (expect {expect})  "
              f"pro={acc_p:.1%} n={n_p}  anti={acc_a:.1%} n={n_a}")

    # --- Hard assertions ---
    g, *_ = compute_diff(score_responses(fake_responses(qs, "correct"), qs)[0])
    assert abs(g) < 0.01, f"always-correct must score Gap~0, got {g}"

    g, *_ = compute_diff(score_responses(fake_responses(qs, "stereo"), qs)[0])
    assert g > 0.95, f"always-stereotype must score Gap~+1, got {g}"

    g, *_ = compute_diff(score_responses(fake_responses(qs, "random", 1), qs)[0])
    assert abs(g) < 0.10, f"random must score Gap~0, got {g}"

    # Position-bias control: a model that always answers 'A' has no
    # stereotype sensitivity at all, so any nonzero Gap it scores is
    # position leakage from correct_letter being correlated with condition.
    # The A2 audit showed a mild imbalance (56.1% vs 49.5%), so allow ~10%.
    g, *_ = compute_diff(score_responses(fake_responses(qs, "always_a"), qs)[0])
    assert abs(g) < 0.10, (
        f"always-'A' model scored Gap={g:+.1%} — correct_letter is correlated "
        f"with pro/anti, so Gap is contaminated by answer position. "
        f"Apply fix C3 (randomize A/B assignment) before collecting new data."
    )
    print("\nAll synthetic assertions passed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_winobias_synthetic.py <winobias_run.json>")
        sys.exit(1)
    main(sys.argv[1])