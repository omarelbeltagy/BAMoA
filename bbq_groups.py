# bbq_groups.py — resolve BBQ answer options to target / non-target groups.
#
# BBQ label vocabularies vary too much for any prefix rule:
#   Age                 old / nonOld              (prefix works)
#   SES                 lowSES / highSES          (no prefix at all)
#   Religion            Muslim / Christian / ...  (no prefix at all)
#   Race_x_gender       F-African American / ...  (compound)
#
# Instead we resolve at the ITEM level. Each item has two named options plus
# `unknown`. `additional_metadata.stereotyped_groups` names the disadvantaged
# group. If exactly one named option matches it, the other is the non-target
# by elimination. Anything else is unresolved and MUST be skipped.
import re


def _norm(s):
    """Lowercase, keep alphanumerics only — tolerant surface matching."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _components(label):
    """Split compound labels ('F-African American') into parts."""
    return [_norm(p) for p in re.split(r"[-_]", str(label)) if _norm(p)]


def _is_negation_of(label_n, target_n):
    """True if label is an explicit negation: nonObese/obese,
    noVisibleDifference/visibleDifference, notPregnant/pregnant."""
    for prefix in ("non", "not", "no"):
        if label_n.startswith(prefix) and label_n[len(prefix):] == target_n:
            return True
    return False


def _matches(label, stereotyped_groups):
    """Does this option label name one of the stereotyped groups?"""
    label_n = _norm(label)
    if not label_n:
        return False
    parts = _components(label)
    for g in stereotyped_groups:
        g_n = _norm(g)
        if not g_n:
            continue
        # A negation is never a match, even though it contains the target
        # as a substring ('nonobese' contains 'obese').
        if _is_negation_of(label_n, g_n):
            continue
        if label_n == g_n:
            return True
        if g_n in parts:          # compound component match
            return True
        if any(_is_negation_of(p, g_n) for p in parts):
            continue
    return False


def resolve_item_targets(answer_info, stereotyped_groups, option_order=None):
    """
    Return {letter: "target" | "nontarget" | "unknown"} or None if undecidable.

    None is a hard signal: exclude the item and count it. Never guess.
    `option_order` maps display slot -> source ans index (fix C3); pass None
    for run files collected before that fix.
    """
    if not stereotyped_groups:
        return None

    result, named = {}, []
    for slot, letter in enumerate("ABC"):
        idx = option_order[slot] if option_order else slot
        entry = answer_info.get(f"ans{idx}")
        label = entry[1] if entry else None
        if not label or _norm(label) == "unknown":
            result[letter] = "unknown"
        else:
            named.append((letter, label))

    if len(named) != 2:
        return None  # malformed item

    matched = [l for l, lab in named if _matches(lab, stereotyped_groups)]
    if len(matched) != 1:
        return None  # zero or ambiguous matches — skip, don't guess

    for letter, _ in named:
        result[letter] = "target" if letter == matched[0] else "nontarget"
    return result