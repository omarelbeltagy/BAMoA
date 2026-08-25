# audits/audit_bbq_labels.py
# Read-only audit: how often does the `startswith("non")` heuristic
# correctly identify the BBQ target group? Run against a saved run file.
import json, sys
from collections import defaultdict

def old_is_target(label):
    return not label.startswith("non")

def main(path):
    with open(path) as f:
        questions = json.load(f)

    labels_by_cat = defaultdict(set)
    for q in questions:
        m = q["bbq_metadata"]
        for key, entry in m["answer_info"].items():
            if entry:
                labels_by_cat[m["category"]].add(entry[1])

    print(f"{'category':<25} {'labels seen':<60} heuristic")
    for cat in sorted(labels_by_cat):
        labels = sorted(l for l in labels_by_cat[cat] if l != "unknown")
        # A category is "heuristic-safe" only if exactly one of the two
        # non-unknown labels starts with 'non'.
        n_non = sum(1 for l in labels if l.startswith("non"))
        verdict = "OK" if n_non == 1 else f"BROKEN (n_non={n_non})"
        print(f"{cat:<25} {str(labels)[:58]:<60} {verdict}")

    # How many questions live in broken categories?
    broken = {c for c, ls in labels_by_cat.items()
              if sum(1 for l in ls if l != "unknown" and l.startswith("non")) != 1}
    n_broken = sum(1 for q in questions if q["bbq_metadata"]["category"] in broken)
    print(f"\n{n_broken}/{len(questions)} questions ({n_broken/len(questions):.1%}) "
          f"are in categories where the heuristic fails.")

if __name__ == "__main__":
    main(sys.argv[1])