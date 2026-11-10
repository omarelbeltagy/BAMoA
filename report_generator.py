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
import json
import os
from datetime import datetime
from collections import defaultdict
from bbq_scorer import (
    score_responses_ambig,
    score_responses_disambig,
    directional_lean,
    bias_score_amb,
    score_gap,
    compute_gap,
    gt_is_stereotype_consistent,
    MIN_VALID_N as BBQ_MIN_VALID_N,
)
from winobias_scorer import (
    score_responses as wb_score_responses,
    accuracy as wb_accuracy,
    compute_diff as wb_compute_diff,
    MIN_VALID_N as WB_MIN_VALID_N,
)
from stats_utils import bootstrap_ci, paired_bootstrap_delta

NULL_RATE_WARNING_THRESHOLD = 0.10  # flag rows with >10% null responses

def fmt_pct(value):
    return f"{value:.1%}" if value is not None else "N/A"


def fmt_signed_pct(value):
    return f"{value:+.1%}" if value is not None else "N/A"


def null_flag(null_rate):
    return " ⚠" if null_rate is not None and null_rate > NULL_RATE_WARNING_THRESHOLD else ""

def find_latest_run(output_dir):
    """Return the most recent run_*.json in output_dir, or None."""
    if not os.path.isdir(output_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(output_dir, "run_*.json")))
    return candidates[-1] if candidates else None


def load_run(path):
    with open(path) as f:
        return json.load(f)

def fmt_ci(point, lo, hi, signed=True, pct=True):
    """Render '−2.3% [−5.1%, +0.4%]'. N/A when the point estimate is None."""
    if point is None:
        return "N/A"
    f = (lambda v: f"{v:+.1%}") if (signed and pct) else \
        (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:+.3f}")
    if lo is None or hi is None:
        return f(point)
    return f"{f(point)} [{f(lo)}, {f(hi)}]"

def make_wb_gap_metric_model(layer, model):
    """Per-model WinoBias Gap, for bootstrap. Per-model n is small, so these
    are the report's noisiest figures and most need intervals."""
    def metric(qs):
        if layer is None:
            responses = [{model: q["final_response"]} for q in qs]
        else:
            responses = [{model: q["layers"][layer].get(model)} for q in qs]
        overall, _ = wb_score_responses(responses, qs)
        return wb_compute_diff(overall)[0]
    return metric

def make_gap_metric(layer):
    """Gap for one layer, as a function of a question list (for bootstrap).
    `layer` is a layer name, or None for the aggregator."""
    def metric(qs):
        if layer is None:
            responses = [{"aggregator": q["final_response"]} for q in qs]
        else:
            responses = [q["layers"][layer] for q in qs]
        overall, _ = score_gap(responses, qs)
        return compute_gap(overall)[0]
    return metric

def wb_noise_floor(qs, n_seeds=200):
    """
    Empirical detectability baselines for WinoBias Gap at this sample size:
      - random: a zero-sensitivity model; the spread is the noise floor
      - always-A: a position-only model; nonzero Gap here is contamination
        from correct_letter correlating with the pro/anti condition
    """
    import random as _r
    from winobias_scorer import parse_answer  # noqa: F401

    def gap_for(answer_fn, seed):
        rng = _r.Random(seed)
        responses = [{"synthetic": answer_fn(q, rng)} for q in qs]
        overall, _ = wb_score_responses(responses, qs)
        return wb_compute_diff(overall)[0]

    gaps = [g for s in range(n_seeds)
            if (g := gap_for(lambda q, rng: rng.choice(["A", "B"]), s)) is not None]
    gaps.sort()
    lo, hi = gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps)) - 1]
    always_a = gap_for(lambda q, rng: "A", 0)

    return lo, hi, always_a


def make_sdis_metric(layer, polarity, is_ambig):
    """s_DIS for one layer, as a function of a question list."""
    scorer = score_responses_ambig if is_ambig else score_responses_disambig
    def metric(qs):
        if layer is None:
            responses = [{"aggregator": q["final_response"]} for q in qs]
        else:
            responses = [q["layers"][layer] for q in qs]
        overall, _, _ = scorer(responses, qs, polarity)
        return directional_lean(overall["stereotyped"],
                                overall["anti_stereotyped"])
    return metric

def bbq_gap_table(qs, layer_names, n_boot=500):
    """Layer-wise Gap with bootstrap CIs, plus a paired L1-vs-aggregator delta."""
    n_res = sum(1 for q in qs if gt_is_stereotype_consistent(q) is not None)
    lines = [
        f"\n**Gap (GT-conditioned)** — resolved: {n_res}/{len(qs)} "
        f"({n_res/len(qs):.1%})\n",
        "| Layer | acc (GT stereo-consistent) | acc (GT stereo-inconsistent) | Gap [95% CI] |",
        "|---|---|---|---|",
    ]

    for layer in list(layer_names) + [None]:
        name = "**final (aggregator)**" if layer is None else layer
        metric = make_gap_metric(layer)
        point, lo, hi = bootstrap_ci(qs, metric, n_boot=n_boot)

        responses = ([{"aggregator": q["final_response"]} for q in qs]
                     if layer is None else [q["layers"][layer] for q in qs])
        overall, _ = score_gap(responses, qs)
        _, acc_c, n_c, acc_a, n_a = compute_gap(overall)

        lines.append(
            f"| {name} | {fmt_pct(acc_c)} (n={n_c}) | {fmt_pct(acc_a)} (n={n_a}) "
            f"| {fmt_ci(point, lo, hi)} |"
        )

    # Paired delta: same items through both layers, so pair the resample.
    d, dlo, dhi = paired_bootstrap_delta(
        qs, make_gap_metric(None), make_gap_metric(layer_names[0]), n_boot=n_boot)
    lines.append(
        f"\n**Aggregator − Layer 1 (paired):** {fmt_ci(d, dlo, dhi)}  "
        f"— CI excluding zero indicates a real change across the stack.\n")
    return "\n".join(lines)

def make_wb_gap_metric(layer):
    def metric(qs):
        if layer is None:
            responses = [{"aggregator": q["final_response"]} for q in qs]
        else:
            responses = [q["layers"][layer] for q in qs]
        overall, _ = wb_score_responses(responses, qs)
        return wb_compute_diff(overall)[0]
    return metric

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
                if row['s_dis'] is not None:
                    p, lo, hi = bootstrap_ci(qs, make_sdis_metric(layer, polarity, is_ambig),
                                             n_boot=500)
                    s_dis_str = fmt_ci(p, lo, hi, pct=False)
                else:
                    s_dis_str = "N/A"
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
            if row['s_dis'] is not None:
                p, lo, hi = bootstrap_ci(qs, make_sdis_metric(None, polarity, is_ambig),
                                         n_boot=500)
                s_dis_str = fmt_ci(p, lo, hi, pct=False)
            else:
                s_dis_str = "N/A"
            if is_ambig:
                s_amb_str = f"{row['s_amb']:.3f}" if row['s_amb'] is not None else "N/A"
                summary_rows.append(f"| **final (aggregator)** | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | {s_dis_str} | {s_amb_str} |")
            else:
                summary_rows.append(f"| **final (aggregator)** | {fmt_pct(acc)} | {fmt_pct(row['null_rate'])}{flag} | {s_dis_str} |")
        lines.append("\n".join(summary_rows))
        if not is_ambig:
            lines.append(bbq_gap_table(qs, layer_names))
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
    lines = ["\n### Type 1 — primary bias signal (`Gap = accuracy_pro − accuracy_anti`)\n"]
    header = "| Layer | accuracy_pro | accuracy_anti | Gap | Null rate |"
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
        p, lo, hi = bootstrap_ci(qs, make_wb_gap_metric(layer), n_boot=500)
        diff_str = fmt_ci(p, lo, hi)
        rows.append(f"| {layer} | {pro_str} | {anti_str} | {diff_str} | {fmt_pct(row['null_rate'])}{flag} |")

        pm_rows = [f"\n**{layer} — per model**\n",
                   "| Model | accuracy_pro | accuracy_anti | Gap | Null |",
                   "|---|---|---|---|---|"]
        for model, conds in sorted(per_model.items()):
            short = model.split("/")[-1]
            m_diff, m_acc_pro, m_n_pro, m_acc_anti, m_n_anti = wb_compute_diff(conds)
            m_null = conds["pro"]["null"] + conds["anti"]["null"]
            pro_str = f"{m_acc_pro:.1%} (n={m_n_pro})" if m_acc_pro is not None else f"N/A (n={m_n_pro})"
            anti_str = f"{m_acc_anti:.1%} (n={m_n_anti})" if m_acc_anti is not None else f"N/A (n={m_n_anti})"
            dmp, mlo, mhi = bootstrap_ci(qs, make_wb_gap_metric_model(layer, model),
                                        n_boot=500)
            diff_str = fmt_ci(dmp, mlo, mhi)
            pm_rows.append(f"| {short} | {pro_str} | {anti_str} | {diff_str} | {m_null} |")
        per_model_sections.append("\n".join(pm_rows))

    # aggregator
    final_responses = [{"aggregator": q["final_response"]} for q in qs]
    overall, per_model = wb_score_responses(final_responses, qs)
    row = wb_layer_row_type1("final (aggregator)", overall)
    flag = null_flag(row["null_rate"])
    pro_str = f"{row['acc_pro']:.1%} (n={row['n_pro']})" if row["acc_pro"] is not None else f"N/A (n={row['n_pro']})"
    anti_str = f"{row['acc_anti']:.1%} (n={row['n_anti']})" if row["acc_anti"] is not None else f"N/A (n={row['n_anti']})"
    p, lo, hi = bootstrap_ci(qs, make_wb_gap_metric(None), n_boot=1000)
    rows.append(f"| **final (aggregator)** | {pro_str} | {anti_str} | "
                f"{fmt_ci(p, lo, hi)} | {fmt_pct(row['null_rate'])}{flag} |")

    lines.append("\n".join(rows))

    # Paired delta: same items pass through every layer, so the resample
    # must be paired — unpaired CIs would badly overstate uncertainty.
    first_layer = layer_names[0]
    d, dlo, dhi = paired_bootstrap_delta(
        qs, make_wb_gap_metric(None), make_wb_gap_metric(first_layer))
    lines.append(f"\n**Aggregator − {first_layer} (paired):** {fmt_ci(d, dlo, dhi)}\n")
    nlo, nhi, na = wb_noise_floor(qs)
    lines.append(
        f"\n> **Detectability baselines at n={len(qs)}:** a zero-sensitivity "
        f"(random) model scores Gap in [{nlo:+.1%}, {nhi:+.1%}] (95% of seeds); "
        f"a position-only ('always A') model scores {na:+.1%}. "
        f"Interpret Gap against these, not against zero.\n")
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