# report_generator.py — builds a combined Markdown report (BBQ + WinoBias)
# from the most recent output file of each dataset.
#
# Usage:
#   python report_generator.py
#   python report_generator.py --bbq outputs/bbq/run_X.json --winobias outputs/winobias/run_Y.json
#
# Output: reports/report_YYYYMMDD_HHMMSS.md
import argparse
import glob
import os
from datetime import datetime
from collections import defaultdict

from bbq_scorer import (
    score_responses_ambig,
    score_responses_disambig,
    directional_lean,
    bias_score_amb,
    MIN_VALID_N as BBQ_MIN_VALID_N,
)
from winobias_scorer import (
    score_responses as wb_score_responses,
    accuracy as wb_accuracy,
    compute_diff as wb_compute_diff,
    MIN_VALID_N as WB_MIN_VALID_N,
)

import json

NULL_RATE_WARNING_THRESHOLD = 0.10  # flag rows with >10% null responses


def find_latest_run(output_dir):
    """Return the most recent run_*.json in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(output_dir, "run_*.json")))
    return candidates[-1] if candidates else None


def load_run(path):
    with open(path) as f:
        return json.load(f)


def fmt_pct(value):
    return f"{value:.1%}" if value is not None else "N/A"


def fmt_signed_pct(value):
    return f"{value:+.1%}" if value is not None else "N/A"


def null_flag(null_rate):
    return " ⚠" if null_rate is not None and null_rate > NULL_RATE_WARNING_THRESHOLD else ""


# ─────────────────────────────────────────────────────────────────
# BBQ report section
# ─────────────────────────────────────────────────────────────────

def bbq_layer_row(name, overall, is_ambig):
    n_s = overall["stereotyped"]
    n_a = overall["anti_stereotyped"]
    n_u = overall["unknown"]
    n_null = overall["null_response"]
    n_valid = n_s + n_a + n_u
    n_attempted = n_valid + n_null
    null_rate = n_null / n_attempted if n_attempted else None
    insufficient = (n_s + n_a) < BBQ_MIN_VALID_N  # s_DIS is over committed answers

    if is_ambig:
        accuracy = n_u / n_valid if n_valid else None
    else:
        accuracy = None  # computed separately from correct/incorrect for disambig

    s_dis = None if insufficient else directional_lean(n_s, n_a)
    s_amb = None if insufficient else (bias_score_amb(n_s, n_a, n_u) if is_ambig else None)

    return {
        "name": name, "n_s": n_s, "n_a": n_a, "n_u": n_u,
        "n_valid": n_valid, "n_null": n_null, "null_rate": null_rate,
        "accuracy": accuracy, "s_dis": s_dis, "s_amb": s_amb,
        "insufficient": insufficient,
    }


def bbq_disambig_accuracy(overall):
    n_correct = overall["correct"]
    n_incorrect = overall["incorrect"]
    n_scored = n_correct + n_incorrect
    return (n_correct / n_scored) if n_scored else None


def bbq_category_table(records, is_ambig):
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r["cls"])

    lines = ["| Category | N | s_DIS | s_AMB |" if is_ambig else "| Category | N | s_DIS |",
             "|---|---|---|---|" if is_ambig else "|---|---|---|"]
    for category in sorted(by_category):
        cls_list = by_category[category]
        n_s = sum(1 for c in cls_list if c == "stereotyped")
        n_a = sum(1 for c in cls_list if c == "anti_stereotyped")
        n_u = sum(1 for c in cls_list if c == "unknown")
        n_valid = n_s + n_a + n_u
        if n_valid < BBQ_MIN_VALID_N:
            row = f"| {category} | {n_valid} | N/A | N/A |" if is_ambig else f"| {category} | {n_valid} | N/A |"
            lines.append(row)
            continue
        s_dis = directional_lean(n_s, n_a)
        if is_ambig:
            s_amb = bias_score_amb(n_s, n_a, n_u)
            lines.append(f"| {category} | {n_valid} | {fmt_signed_pct(s_dis) if s_dis is None else f'{s_dis:.3f}'} | {'N/A' if s_amb is None else f'{s_amb:.3f}'} |")
        else:
            lines.append(f"| {category} | {n_valid} | {s_dis:.3f} |")
    return "\n".join(lines)


def generate_bbq_section(path):
    questions = load_run(path)
    lines = [f"## BBQ\n", f"**Source:** `{path}`  ", f"**Total questions:** {len(questions)}\n"]

    layer_names = list(questions[0]["layers"].keys())
    groups = defaultdict(list)
    for q in questions:
        key = (q["bbq_metadata"]["context_condition"], q["bbq_metadata"]["question_polarity"])
        groups[key].append(q)

    for (condition, polarity), qs in sorted(groups.items()):
        is_ambig = condition == "ambig"
        lines.append(f"\n### context={condition}, polarity={polarity}  ({len(qs)} questions)\n")

        summary_header = (
            "| Layer | Accuracy | Null rate | s_DIS | s_AMB |"
            if is_ambig else
            "| Layer | Accuracy | Null rate | s_DIS |"
        )
        summary_sep = "|---|---|---|---|---|" if is_ambig else "|---|---|---|---|"
        summary_rows = [summary_header, summary_sep]

        per_model_sections = []
        category_sections = []

        for layer in layer_names:
            responses = [q["layers"][layer] for q in qs]
            if is_ambig:
                overall, per_model, records = score_responses_ambig(responses, qs, polarity)
            else:
                overall, per_model, records = score_responses_disambig(responses, qs, polarity)

            row = bbq_layer_row(layer, overall, is_ambig)
            acc = row["accuracy"] if is_ambig else bbq_disambig_accuracy(overall)
            flag = null_flag(row["null_rate"])

            if row["insufficient"]:
                if is_ambig:
                    summary_rows.append(f"| {layer} | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | N/A (n={row['n_valid']}) | N/A |")
                else:
                    summary_rows.append(f"| {layer} | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | N/A (n={row['n_valid']}) |")
            else:
                s_dis_str = f"{row['s_dis']:.3f}" if row['s_dis'] is not None else "N/A"
                if is_ambig:
                    s_amb_str = f"{row['s_amb']:.3f}" if row['s_amb'] is not None else "N/A"
                    summary_rows.append(f"| {layer} | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | {s_dis_str} | {s_amb_str} |")
                else:
                    summary_rows.append(f"| {layer} | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | {s_dis_str} |")

            # per-model table for this layer
            pm_header = (
                "| Model | S | A | U | Null | Null rate | s_DIS | s_AMB |"
                if is_ambig else
                "| Model | S | A | U | Null | Accuracy | s_DIS |"
            )
            pm_sep = "|---|---|---|---|---|---|---|---|" if is_ambig else "|---|---|---|---|---|---|---|"
            pm_rows = [f"\n**{layer} — per model**\n", pm_header, pm_sep]
            for model, counts in sorted(per_model.items()):
                short = model.split("/")[-1]
                n_s, n_a, n_u = counts["stereotyped"], counts["anti_stereotyped"], counts["unknown"]
                n_null = counts["null_response"]
                n_valid = n_s + n_a + n_u
                n_attempted = n_valid + n_null
                m_null_rate = n_null / n_attempted if n_attempted else None
                m_flag = null_flag(m_null_rate)
                if n_valid < BBQ_MIN_VALID_N:
                    if is_ambig:
                        pm_rows.append(f"| {short} | {n_s} | {n_a} | {n_u} | {n_null}{m_flag} | {fmt_pct(m_null_rate)} | N/A | N/A |")
                    else:
                        m_acc = (counts["correct"] / (counts["correct"] + counts["incorrect"])) if (counts["correct"] + counts["incorrect"]) else None
                        pm_rows.append(f"| {short} | {n_s} | {n_a} | {n_u} | {n_null}{m_flag} | {fmt_pct(m_acc)} | N/A |")
                else:
                    m_s_dis = directional_lean(n_s, n_a)
                    m_s_dis_str = f"{m_s_dis:.3f}" if m_s_dis is not None else "N/A"
                    if is_ambig:
                        m_s_amb = bias_score_amb(n_s, n_a, n_u)
                        m_s_amb_str = f"{m_s_amb:.3f}" if m_s_amb is not None else "N/A"
                        pm_rows.append(f"| {short} | {n_s} | {n_a} | {n_u} | {n_null}{m_flag} | {fmt_pct(m_null_rate)} | {m_s_dis_str} | {m_s_amb_str} |")
                    else:
                        m_acc = (counts["correct"] / (counts["correct"] + counts["incorrect"])) if (counts["correct"] + counts["incorrect"]) else None
                        pm_rows.append(f"| {short} | {n_s} | {n_a} | {n_u} | {n_null}{m_flag} | {fmt_pct(m_acc)} | {m_s_dis_str} |")
            per_model_sections.append("\n".join(pm_rows))

            category_sections.append(f"\n**{layer} — per category**\n\n" + bbq_category_table(records, is_ambig))

        # aggregator
        final_responses = [{"aggregator": q["final_response"]} for q in qs]
        if is_ambig:
            overall, _, records = score_responses_ambig(final_responses, qs, polarity)
        else:
            overall, _, records = score_responses_disambig(final_responses, qs, polarity)
        row = bbq_layer_row("final (aggregator)", overall, is_ambig)
        acc = row["accuracy"] if is_ambig else bbq_disambig_accuracy(overall)
        flag = null_flag(row["null_rate"])
        if row["insufficient"]:
            if is_ambig:
                summary_rows.append(f"| **final (aggregator)** | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | N/A (n={row['n_valid']}) | N/A |")
            else:
                summary_rows.append(f"| **final (aggregator)** | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | N/A (n={row['n_valid']}) |")
        else:
            s_dis_str = f"{row['s_dis']:.3f}" if row['s_dis'] is not None else "N/A"
            if is_ambig:
                s_amb_str = f"{row['s_amb']:.3f}" if row['s_amb'] is not None else "N/A"
                summary_rows.append(f"| **final (aggregator)** | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | {s_dis_str} | {s_amb_str} |")
            else:
                summary_rows.append(f"| **final (aggregator)** | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | {s_dis_str} |")
        lines.append("\n".join(summary_rows))
        lines.append("\n".join(per_model_sections))
        lines.append("\n".join(category_sections))

    lines.append(f"\n> ⚠ = null rate above {NULL_RATE_WARNING_THRESHOLD:.0%}. "
                  f"N/A = fewer than {BBQ_MIN_VALID_N} valid (non-null) answers.\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# WinoBias report section
# ─────────────────────────────────────────────────────────────────

def wb_layer_row_type1(name, overall):
    diff, acc_pro, n_pro, acc_anti, n_anti = wb_compute_diff(overall)
    n_null = overall["pro"]["null"] + overall["anti"]["null"]
    n_valid = n_pro + n_anti
    n_attempted = n_valid + n_null
    null_rate = n_null / n_attempted if n_attempted else None
    return {
        "name": name, "diff": diff, "acc_pro": acc_pro, "n_pro": n_pro,
        "acc_anti": acc_anti, "n_anti": n_anti, "n_null": n_null,
        "null_rate": null_rate,
    }


def generate_winobias_type1_section(qs, layer_names):
    lines = ["\n### Type 1 — primary bias signal (`Diff = accuracy_pro − accuracy_anti`)\n"]
    header = "| Layer | accuracy_pro | accuracy_anti | Diff | Null rate |"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]

    per_model_sections = []

    for layer in layer_names:
        responses = [q["layers"][layer] for q in qs]
        overall, per_model = wb_score_responses(responses, qs)
        row = wb_layer_row_type1(layer, overall)
        flag = null_flag(row["null_rate"])
        pro_str = fmt_pct(row["acc_pro"]) + f" (n={row['n_pro']})" if row["acc_pro"] is None else f"{row['acc_pro']:.1%} (n={row['n_pro']})"
        anti_str = fmt_pct(row["acc_anti"]) + f" (n={row['n_anti']})" if row["acc_anti"] is None else f"{row['acc_anti']:.1%} (n={row['n_anti']})"
        diff_str = fmt_signed_pct(row["diff"])
        rows.append(f"| {layer} | {pro_str} | {anti_str} | {diff_str} | {fmt_pct(row['null_rate'])}{flag} |")

        pm_rows = [f"\n**{layer} — per model**\n",
                   "| Model | accuracy_pro | accuracy_anti | Diff | Null |",
                   "|---|---|---|---|---|"]
        for model, conds in sorted(per_model.items()):
            short = model.split("/")[-1]
            m_diff, m_acc_pro, m_n_pro, m_acc_anti, m_n_anti = wb_compute_diff(conds)
            m_null = conds["pro"]["null"] + conds["anti"]["null"]
            pro_str = f"{m_acc_pro:.1%} (n={m_n_pro})" if m_acc_pro is not None else f"N/A (n={m_n_pro})"
            anti_str = f"{m_acc_anti:.1%} (n={m_n_anti})" if m_acc_anti is not None else f"N/A (n={m_n_anti})"
            diff_str = fmt_signed_pct(m_diff)
            pm_rows.append(f"| {short} | {pro_str} | {anti_str} | {diff_str} | {m_null} |")
        per_model_sections.append("\n".join(pm_rows))

    # aggregator
    final_responses = [{"aggregator": q["final_response"]} for q in qs]
    overall, per_model = wb_score_responses(final_responses, qs)
    row = wb_layer_row_type1("final (aggregator)", overall)
    flag = null_flag(row["null_rate"])
    pro_str = f"{row['acc_pro']:.1%} (n={row['n_pro']})" if row["acc_pro"] is not None else f"N/A (n={row['n_pro']})"
    anti_str = f"{row['acc_anti']:.1%} (n={row['n_anti']})" if row["acc_anti"] is not None else f"N/A (n={row['n_anti']})"
    rows.append(f"| **final (aggregator)** | {pro_str} | {anti_str} | {fmt_signed_pct(row['diff'])} | {fmt_pct(row['null_rate'])}{flag} |")

    lines.append("\n".join(rows))
    lines.append("\n".join(per_model_sections))
    return "\n".join(lines)


def generate_winobias_type2_section(qs, layer_names):
    lines = ["\n### Type 2 — sanity check only (NOT a bias metric)\n"]
    header = "| Layer | Accuracy (combined) | accuracy_pro | accuracy_anti |"
    sep = "|---|---|---|---|"
    rows = [header, sep]

    for layer in layer_names:
        responses = [q["layers"][layer] for q in qs]
        overall, per_model = wb_score_responses(responses, qs)
        acc_pro, n_pro = wb_accuracy(overall["pro"])
        acc_anti, n_anti = wb_accuracy(overall["anti"])
        combined = defaultdict(int)
        for cond in ("pro", "anti"):
            for k in ("correct", "incorrect", "null"):
                combined[k] += overall[cond][k]
        acc_combined, n_combined = wb_accuracy(combined)
        rows.append(
            f"| {layer} | {fmt_pct(acc_combined)} (n={n_combined}) "
            f"| {fmt_pct(acc_pro)} (n={n_pro}) | {fmt_pct(acc_anti)} (n={n_anti}) |"
        )

    final_responses = [{"aggregator": q["final_response"]} for q in qs]
    overall, _ = wb_score_responses(final_responses, qs)
    acc_pro, n_pro = wb_accuracy(overall["pro"])
    acc_anti, n_anti = wb_accuracy(overall["anti"])
    combined = defaultdict(int)
    for cond in ("pro", "anti"):
        for k in ("correct", "incorrect", "null"):
            combined[k] += overall[cond][k]
    acc_combined, n_combined = wb_accuracy(combined)
    rows.append(
        f"| **final (aggregator)** | {fmt_pct(acc_combined)} (n={n_combined}) "
        f"| {fmt_pct(acc_pro)} (n={n_pro}) | {fmt_pct(acc_anti)} (n={n_anti}) |"
    )

    lines.append("\n".join(rows))
    return "\n".join(lines)


def generate_winobias_section(path):
    questions = load_run(path)
    lines = [f"## WinoBias\n", f"**Source:** `{path}`  ", f"**Total questions:** {len(questions)}\n"]

    type1_qs = [q for q in questions if q["winobias_metadata"]["wb_type"] == "type1"]
    type2_qs = [q for q in questions if q["winobias_metadata"]["wb_type"] == "type2"]
    lines.append(f"Type 1: {len(type1_qs)} questions  \nType 2: {len(type2_qs)} questions\n")

    if not questions:
        return "\n".join(lines) + "\n\n_No questions in file._\n"
    layer_names = list(questions[0]["layers"].keys())

    if type1_qs:
        lines.append(generate_winobias_type1_section(type1_qs, layer_names))
    else:
        lines.append("\n_No Type 1 data found._\n")

    if type2_qs:
        lines.append(generate_winobias_type2_section(type2_qs, layer_names))

    lines.append(f"\n> ⚠ = null rate above {NULL_RATE_WARNING_THRESHOLD:.0%}. "
                  f"N/A = fewer than {WB_MIN_VALID_N} valid (non-null) answers.\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a combined BBQ + WinoBias report.")
    parser.add_argument("--bbq", metavar="PATH", default=None,
                         help="BBQ run file to use (default: latest in outputs/bbq/)")
    parser.add_argument("--winobias", metavar="PATH", default=None,
                         help="WinoBias run file to use (default: latest in outputs/winobias/)")
    args = parser.parse_args()

    bbq_path = args.bbq or find_latest_run("outputs/bbq")
    winobias_path = args.winobias or find_latest_run("outputs/winobias")

    if not bbq_path and not winobias_path:
        print("No output files found for either dataset.")
        print("Nothing to report. Run one of the following first:")
        print("  python app.py --dataset bbq")
        print("  python app.py --dataset winobias")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    out_path = f"reports/report_{timestamp}.md"

    sections = [f"# BA-MoA Report\n\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n"]

    if bbq_path:
        print(f"Generating BBQ section from {bbq_path} ...")
        sections.append(generate_bbq_section(bbq_path))
    else:
        print("WARNING: no output file found for dataset 'bbq' — skipping BBQ section. "
              "Run: python app.py --dataset bbq")
        sections.append("## BBQ\n\n_No output file found for this dataset. "
                         "Run `python app.py --dataset bbq` first._\n")

    if winobias_path:
        print(f"Generating WinoBias section from {winobias_path} ...")
        sections.append(generate_winobias_section(winobias_path))
    else:
        print("WARNING: no output file found for dataset 'winobias' — skipping WinoBias section. "
              "Run: python app.py --dataset winobias")
        sections.append("## WinoBias\n\n_No output file found for this dataset. "
                         "Run `python app.py --dataset winobias` first._\n")

    with open(out_path, "w") as f:
        f.write("\n\n".join(sections))

    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()