# test_bbq_scorer_synthetic.py — a scorer that passes these is measuring
# stereotype alignment; one that fails is measuring something else.
import json, sys, random
from bbq_scorer import score_gap, compute_gap, gt_is_stereotype_consistent

def _other(m):
    return next(l for l in "ABC" if l != m["correct_answer"])

def fake_responses(questions, mode, seed=0):
    rng = random.Random(seed)
    out = []
    for q in questions:
        m = q["bbq_metadata"]
        if mode == "correct":
            ans = m["correct_answer"]
        elif mode == "stereo":
            consistent = gt_is_stereotype_consistent(q)
            ans = (m["correct_answer"] if consistent in (True, None)
                   else _other(m))
        else:
            ans = rng.choice(["A", "B", "C"])
        out.append({"fake": ans})
    return out

def main(path):
    with open(path) as f:
        qs = [q for q in json.load(f)
              if q["bbq_metadata"]["context_condition"] == "disambig"]
    print(f"{len(qs)} disambiguated questions\n")

    for mode, expect in [("correct", "~0"), ("stereo", "large +"), ("random", "~0")]:
        overall, _ = score_gap(fake_responses(qs, mode), qs)
        gap, ac, nc, aa, na = compute_gap(overall)
        print(f"{mode:<8} Gap={gap:+.3f}  (expect {expect})  "
              f"acc_stereo={ac:.1%} n={nc}  acc_anti={aa:.1%} n={na}")

    g, *_ = compute_gap(score_gap(fake_responses(qs, "correct"), qs)[0])
    assert abs(g) < 0.01, f"always-correct must score ~0, got {g}"
    g, *_ = compute_gap(score_gap(fake_responses(qs, "random", 1), qs)[0])
    assert abs(g) < 0.10, f"random must score ~0, got {g}"
    g, *_ = compute_gap(score_gap(fake_responses(qs, "stereo"), qs)[0])
    assert g > 0.80, f"always-stereotype must score >>0, got {g}"
    print("\nAll synthetic assertions passed.")

if __name__ == "__main__":
    main(sys.argv[1])