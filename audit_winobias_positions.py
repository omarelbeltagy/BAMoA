# audits/audit_winobias_positions.py
# Read-only audit: is WinoBias Diff confounded with answer position?
import json, sys
from collections import defaultdict
from winobias_scorer import parse_answer

def main(path):
    with open(path) as f:
        questions = [q for q in json.load(f)
                     if q["winobias_metadata"]["wb_type"] == "type1"]

    # 1. Is correct_letter balanced across conditions?
    dist = defaultdict(lambda: defaultdict(int))
    for q in questions:
        m = q["winobias_metadata"]
        dist[m["wb_condition"]][m["correct_letter"]] += 1
    print("correct_letter distribution by condition:")
    for cond in ("pro", "anti"):
        a, b = dist[cond]["A"], dist[cond]["B"]
        print(f"  {cond:<5} A={a} B={b}  P(A)={a/(a+b):.1%}" if a + b else f"  {cond}: no data")

    layers = list(questions[0]["layers"].keys())

    # 2. Per-model raw answer preference (position bias, independent of correctness)
    print("\nP(answer='A') per model, layer_1:")
    pref = defaultdict(lambda: [0, 0])
    for q in questions:
        for model, text in q["layers"][layers[0]].items():
            letter = parse_answer(text)
            if letter:
                pref[model][0 if letter == "A" else 1] += 1
    for model, (a, b) in sorted(pref.items()):
        print(f"  {model.split('/')[-1]:<32} P(A)={a/(a+b):.1%}  (n={a+b})")

    # 3. Diff recomputed WITHIN each correct_letter stratum.
    #    If the gap is real, it survives in both strata.
    print("\nDiff stratified by correct_letter (layer_1):")
    for stratum in ("A", "B"):
        acc = defaultdict(lambda: [0, 0])  # cond -> [correct, total]
        for q in questions:
            m = q["winobias_metadata"]
            if m["correct_letter"] != stratum:
                continue
            for text in q["layers"][layers[0]].values():
                letter = parse_answer(text)
                if not letter:
                    continue
                acc[m["wb_condition"]][1] += 1
                if letter == m["correct_letter"]:
                    acc[m["wb_condition"]][0] += 1
        p, a_ = acc["pro"], acc["anti"]
        if p[1] and a_[1]:
            diff = p[0]/p[1] - a_[0]/a_[1]
            print(f"  correct={stratum}: pro={p[0]/p[1]:.1%} (n={p[1]})  "
                  f"anti={a_[0]/a_[1]:.1%} (n={a_[1]})  Diff={diff:+.1%}")

if __name__ == "__main__":
    main(sys.argv[1])