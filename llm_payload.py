"""
Evidence payload for the report LLM.

The verdict dict from `clickhouse_queries.build_verdict` is built for the UI: it
carries the full dispersion `ranking` (one row per dimension, five floats each),
day-by-day window lists, and duplicated labels. Serialised with indent=2 that is
roughly 1,200-1,500 characters of which the model needs maybe a quarter.

This module maps the verdict onto exactly the field names the report prompt
refers to, and nothing else. Rules followed:

* Nothing is invented. Every number here is either copied from the verdict or is
  a rounding / unit change of one (ratio -> delta_pct). Fields with no source
  are omitted rather than filled with a placeholder, so the model cannot quote
  a number that no query produced.
* Ratio metrics come from sum/sum, per the dataset glossary, which states that
  fill rate, render rate, CTR, eCPM and RPR must be computed as sum over sum for
  a group and never as an average of per-row or per-day ratios.
* Keys keep the exact names used in the prompt. Shortening them would save ~20
  tokens and cost a prompt rewrite plus a debugging headache.
"""

import json


# ---------------------------------------------------------------------------
# Evidence construction
# ---------------------------------------------------------------------------

def _pct(ratio):
    """ratio (1.0 = unchanged) -> signed percentage change."""
    if ratio is None:
        return None
    return round((float(ratio) - 1.0) * 100, 1)


def _window_label(window):
    """['2026-05-04','2026-05-05',...] -> '2026-05-04..2026-05-06'.

    A five-day incident costs 5 quoted date strings as a list and one range
    string here. The model only ever restates the span.
    """
    if not window:
        return None
    return window[0] if len(window) == 1 else f"{window[0]}..{window[-1]}"


def build_evidence(verdict: dict, headline: dict | None = None,
                   seasonality_note: str | None = None) -> dict:
    """Map a UI verdict onto the minimal evidence the report prompt needs.

    `headline` is the platform-level move for the driving metric, from
    clickhouse_queries.incident_headline(). Without it, current_value /
    baseline_value / delta_pct are omitted entirely — the prompt then has no
    numbers to quote for the top-level summary, which is the correct failure
    mode. Silently substituting the culprit segment's numbers would make the
    model state a platform figure that is actually a segment figure.
    """
    ev = {
        "window": _window_label(verdict.get("window", [])),
        "driving_metric": verdict.get("metric_label") or verdict.get("metric"),
        "has_culprit": bool(verdict.get("has_culprit")),
    }

    if headline:
        ev["current_value"] = round(float(headline["current_value"]), 4)
        ev["baseline_value"] = round(float(headline["baseline_value"]), 4)
        ev["delta_pct"] = round(float(headline["delta_pct"]), 1)

    if verdict.get("has_culprit"):
        ev["culprit_dimension"] = verdict.get("culprit_dimension")
        ev["culprit_value"] = verdict.get("culprit_value")
        ev["culprit_delta_pct"] = _pct(verdict.get("ratio"))
    else:
        ev["note"] = verdict.get(
            "note", "uniform movement across all dimensions — no single segment responsible"
        )

    ev["ruled_out_dimensions"] = verdict.get("ruled_out", [])

    if seasonality_note:
        ev["seasonality_note"] = seasonality_note

    return ev


# ---------------------------------------------------------------------------
# Serialisation formats
#
# The model is asked to RETURN json; it does not have to RECEIVE json. Three
# options, cheapest last. Whichever you pick, keep it consistent — the prompt
# describes the input format, so switching means editing one line of the prompt.
# ---------------------------------------------------------------------------

def as_json_pretty(ev: dict) -> str:
    """The original format. Kept for comparison only."""
    return json.dumps(ev, indent=2, default=str, ensure_ascii=False)


def as_json_compact(ev: dict) -> str:
    """Same structure, no whitespace. Recommended default.

    Keeps the prompt's 'you will receive a JSON verdict' phrasing honest and
    removes all ambiguity about types, at a small cost over key:value lines.

    ensure_ascii=False matters here: an em-dash escapes to \\u2014, six
    characters for one, and the no-culprit note contains one.
    """
    return json.dumps(ev, separators=(",", ":"), default=str, ensure_ascii=False)


def as_kv_lines(ev: dict) -> str:
    """One `key: value` per line. Cheapest of the three.

    Drops braces, quotes and most commas. Models parse this reliably, but types
    become implicit — 'true' and 'Android 15' are both bare strings — so only
    use it if you have checked your model handles it. Requires changing the
    prompt's description of the input from JSON to key/value lines.
    """
    lines = []
    for k, v in ev.items():
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        elif isinstance(v, bool):
            v = "yes" if v else "no"
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


FORMATS = {
    "json_pretty": as_json_pretty,
    "json_compact": as_json_compact,
    "kv_lines": as_kv_lines,
}


def serialise(ev: dict, fmt: str = "json_compact") -> str:
    return FORMATS[fmt](ev)