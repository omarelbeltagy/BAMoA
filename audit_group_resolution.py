# audit_group_resolution.py — read-only: how well does the resolver cover
# each BBQ category? Run before wiring bbq_groups into the scorer.
import json, sys
from collections import defaultdict
from bbq_groups import resolve_item_targets

def main(path):
    with open(path) as f:
        questions = json.load(f)

    stats = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)

    for q in questions:
        m = q["bbq_metadata"]
        sg = (m.get("additional_metadata") or {}).get("stereotyped_groups")
        res, tier = resolve_item_targets(m["answer_info"], sg,
                                         m.get("option_order"),
                                         category=m["category"],
                                         return_tier=True)
        cat = m["category"]
        if res is None:
            stats[cat]["fail"] += 1
            if len(examples[cat]) < 3:
                labels = [(m["answer_info"].get(f"ans{i}") or [None, None])[1]
                          for i in range(3)]
                examples[cat].append({"sg": sg, "labels": labels})
        else:
            stats[cat]["ok"] += 1
            stats[cat][f"tier{tier}"] += 1

    print(f"{'category':<25} {'t1':>6} {'t2':>6} {'t3':>6} "
          f"{'unres':>7} {'rate':>8}")
    tot_ok = tot_fail = 0
    tot = defaultdict(int)
    for cat in sorted(stats):
        ok, fail = stats[cat]["ok"], stats[cat]["fail"]
        tot_ok += ok; tot_fail += fail
        for t in ("tier1", "tier2", "tier3"):
            tot[t] += stats[cat][t]
        print(f"{cat:<25} {stats[cat]['tier1']:>6} {stats[cat]['tier2']:>6} "
              f"{stats[cat]['tier3']:>6} {fail:>7} {ok/(ok+fail):>7.1%}")
    print(f"{'TOTAL':<25} {tot['tier1']:>6} {tot['tier2']:>6} "
          f"{tot['tier3']:>6} {tot_fail:>7} "
          f"{tot_ok/(tot_ok+tot_fail):>7.1%}")

    if any(examples.values()):
        print("\n--- unresolved examples (for alias tuning) ---")
        for cat, exs in sorted(examples.items()):
            print(f"\n{cat}:")
            for e in exs:
                print(f"  stereotyped_groups={e['sg']}")
                print(f"  answer labels      ={e['labels']}")

if __name__ == "__main__":
    main(sys.argv[1])