"""
Drill-down panels.

Every function here answers "why?" for something already on screen. None of it
feeds detection or the verdict — app.py's pipeline is unchanged.

Things worth knowing before editing this file:

1. Caching is not optional any more. Click-to-drill means Streamlit reruns the
   whole script on every interaction, so an uncached ClickHouse call would fire
   again on each click. Every query below goes through @st.cache_data. The
   leading `_client` argument is underscored so Streamlit skips hashing it.

2. Click selection needs Streamlit >= 1.49 (requirements.txt pins this).
   Scatter, bar and dataframe selection are used below. Pie selection is
   deliberately NOT used — it is unreliable across versions — so the metric
   filter is a radio instead.

3. Every function takes a `grain` (see granularity.py) instead of a bare
   table name. A grain bundles the table AND the time column name/type, so
   the same function works against the daily or the hourly table without an
   if/else anywhere in this file. The resulting DataFrames always use a
   column named `t` for the time axis regardless of grain — see
   clickhouse_queries.py's docstring for why.

4. Every cached_* function below is wrapped in a Langfuse span (see
   langfuse_tracing.py). Because it only runs on a genuine cache miss, a span
   here means "a real ClickHouse call happened" — not "the user saw this,"
   which is a different and looser guarantee. See that module's docstring for
   the reasoning.
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import clickhouse_queries as chq
import langfuse_tracing as lft

CACHE_TTL = 600  # seconds; drill-downs re-query often, the agg table moves


# ---------------------------------------------------------------------------
# Cached query wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_dimension_values(_client, metric, window_values, dimension_name, grain):
    with lft.traced("drill.dimension_values",
                    input={"metric": metric, "dimension": dimension_name,
                           "window": [str(v) for v in window_values], "grain": grain.key}) as span:
        df = chq.drill_dimension_values(_client, metric, list(window_values), dimension_name, grain)
        span.update(output=lft.summarize_df(df))
        return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_day_top_movers(_client, metric, bucket_value, grain, limit=15):
    with lft.traced("drill.day_top_movers",
                    input={"metric": metric, "bucket": str(bucket_value), "grain": grain.key}) as span:
        df = chq.drill_day_top_movers(_client, metric, bucket_value, grain, limit)
        span.update(output=lft.summarize_df(df))
        return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_segment_timeseries(_client, metric, dimension_name, dimension_value, grain):
    with lft.traced("drill.segment_timeseries",
                    input={"metric": metric, "dimension": dimension_name,
                           "value": dimension_value, "grain": grain.key}) as span:
        df = chq.drill_segment_timeseries(_client, metric, dimension_name, dimension_value, grain)
        span.update(output=lft.summarize_df(df))
        return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_revenue_decomposition(_client, window_values, grain):
    with lft.traced("drill.revenue_decomposition",
                    input={"window": [str(v) for v in window_values], "grain": grain.key}) as span:
        df = chq.drill_revenue_decomposition(_client, list(window_values), grain)
        span.update(output=lft.summarize_df(df))
        return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_onset_heatmap(_client, metric, window_values, dimension_name, grain, top_n=15):
    with lft.traced("drill.onset_heatmap",
                    input={"metric": metric, "dimension": dimension_name,
                           "window": [str(v) for v in window_values], "grain": grain.key}) as span:
        df = chq.drill_onset_heatmap(_client, metric, list(window_values), dimension_name, grain, top_n)
        span.update(output=lft.summarize_df(df))
        return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_headline(_client, metric, window_values, grain):
    with lft.traced("drill.incident_headline",
                    input={"metric": metric, "window": [str(v) for v in window_values], "grain": grain.key}) as span:
        result = chq.incident_headline(_client, metric, list(window_values), grain)
        span.update(output=result)
        return result


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_report(evidence_key: str, _evidence: dict, model: str | None = None) -> dict:
    """Cache the report on the evidence, not the verdict.

    Wrapped in traced_root (not traced()) so the Groq generation gets its own
    session-scoped parent to nest under — the langfuse.openai drop-in's
    ambient get_client() pickup would otherwise make this call its own
    disconnected root trace, outside the "one session groups everything"
    story the rest of the app follows. Because this only executes on a
    genuine cache miss, the trace id/url are computed once and cached inside
    the report dict — a cache hit correctly reuses the same link rather than
    fabricating a new, empty trace.
    """
    from llm_stub import generate_llm_report
    with lft.traced_root("incident-report", input={"has_evidence": True}) as span:
        report = generate_llm_report(_evidence, model=model)
        ids = lft.current_ids()
        span.update(output={"has_culprit": _evidence.get("has_culprit")})
    report["_trace_id"] = ids.get("trace_id")
    report["_trace_url"] = ids.get("trace_url")
    return report


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_diagnosis(verdict_key: str, _verdict: dict) -> str:
    """Cache the LLM diagnosis on the verdict's content. Legacy string-return
    path; generate_llm_report/render_report is the current one."""
    from llm_stub import generate_llm_diagnosis
    return generate_llm_diagnosis(_verdict)


def verdict_key(verdict: dict) -> str:
    return json.dumps(verdict, sort_keys=True, default=str)


REPORT_SECTIONS = [
    ("top_level_summary", "What moved"),
    ("root_cause_localization", "Where it localises"),
    ("checked_and_ruled_out", "Checked and cleared"),
]


def render_report(client, verdict: dict, grain, key_prefix: str):
    """Build the trimmed evidence payload, generate the report, render it."""
    import llm_payload as lp

    window_values = _parse_window(verdict["window"], grain)

    headline, note = None, None
    try:
        headline = cached_headline(client, verdict["metric"], tuple(window_values), grain)
        note = chq.seasonality_note(headline, grain)
    except Exception as e:
        st.caption(f"Headline query unavailable ({safe_error(e)}); the report will omit platform-level figures.")

    evidence = lp.build_evidence(verdict, headline or None, note)
    # model is part of the cache key: switching models re-generates the report
    chosen_model = st.session_state.get("groq_model")
    report = cached_report(
        json.dumps(evidence, sort_keys=True, default=str), evidence, chosen_model
    )

    for key, label in REPORT_SECTIONS:
        st.markdown(f"**{label}**")
        st.info(report.get(key, "—"))

    usage = report.get("_usage")
    if usage:
        st.caption(
            f"Groq usage — prompt: {usage['prompt_tokens']} · completion: "
            f"{usage['completion_tokens']} · total: {usage['total_tokens']} tokens "
            f"(`{usage['model']}`). This is what Groq actually billed, not an estimate."
        )
    if report.get("_trace_url"):
        st.caption(
            f"🔗 [View this report's trace in Langfuse]({report['_trace_url']}) · "
            f"`trace_id={report.get('_trace_id')}` — captured when this report was "
            "generated; a cached hit reuses the same link, not a new trace."
        )
    elif not report.get("_trace_id"):
        st.caption("No Langfuse trace for this report — tracing was off when it was generated.")

    with st.expander("Evidence sent to the model", expanded=False):
        fmt = st.radio(
            "Payload format", list(lp.FORMATS.keys()), index=1,
            horizontal=True, key=f"{key_prefix}_fmt",
        )
        payload = lp.serialise(evidence, fmt)
        st.code(payload, language="json" if fmt.startswith("json") else "text")
        st.caption(
            f"{len(payload)} characters. The dispersion ranking, per-bucket window list and "
            "duplicated labels are all dropped — the model only receives what the report "
            "is allowed to quote."
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_window(window_strs, grain):
    """The verdict's `window` list (plain strings) back into date/datetime
    objects matching the grain, for re-querying drill-downs."""
    if grain.is_datetime:
        return [pd.to_datetime(s).to_pydatetime() for s in window_strs]
    return [pd.to_datetime(s).date() for s in window_strs]


_REDACT_KEYS = (
    "CLICKHOUSE_HOST", "CLICKHOUSE_DATABASE",
    "CLICKHOUSE_TABLE_DAILY", "CLICKHOUSE_TABLE_HOURLY",
    "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD",
)


def safe_error(exc) -> str:
    """Exception text with connection details stripped out.

    ClickHouse driver errors embed the full host and echo the failing SQL, which
    means the table name, the database and the column layout all end up on
    screen. This keeps the message useful for debugging without publishing the
    data source.
    """
    import os
    import re

    text = str(exc)
    for key in _REDACT_KEYS:
        value = os.environ.get(key)
        if value and len(value) > 3:
            text = text.replace(value, "<redacted>")
    # host:port patterns the env lookup may have missed
    text = re.sub(r"https?://[^\s'\"]+", "<redacted>", text)
    text = re.sub(r"\b[\w.-]+\.(?:clickhouse\.cloud|amazonaws\.com)\b(?::\d+)?",
                  "<redacted>", text)
    # SQL echoed back by the server
    if "DB::Exception" in text:
        text = text.split(" in scope ")[0]
    return text[:400]


def _selected_points(event):
    """Normalise a Streamlit plotly selection event into a list of points."""
    if not event:
        return []
    selection = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if not selection:
        return []
    return selection.get("points", []) if isinstance(selection, dict) else getattr(selection, "points", [])


def _selected_rows(event):
    if not event:
        return []
    selection = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if not selection:
        return []
    return selection.get("rows", []) if isinstance(selection, dict) else getattr(selection, "rows", [])


# ---------------------------------------------------------------------------
# Drill-down 1 — single bucket (a day, or an hour), from a click on the trend chart
# ---------------------------------------------------------------------------

def render_day_panel(client, metric: str, bucket_value, grain, key_prefix: str):
    st.markdown(f"#### 🔎 {chq.METRIC_LABELS[metric]} at `{bucket_value}`")
    st.caption(
        "Biggest movers in this one bucket, across every dimension at once. Ranked by "
        "contribution to the total change, not by raw ratio — so tiny segments with "
        "wild ratios stay out of the way. The same underlying shift can appear under "
        "more than one dimension."
    )

    with st.spinner("Querying detail..."):
        try:
            movers = cached_day_top_movers(client, metric, bucket_value, grain)
        except Exception as e:
            st.warning(f"Drill-down failed: {safe_error(e)}")
            return

    if movers.empty:
        st.info("No per-dimension rows for this bucket.")
        return

    display = movers.copy()
    display["direction"] = ["🔻" if r < 1 else "🔺" for r in display["ratio"].fillna(1)]
    display = display[[
        "direction", "dimension_name", "dimension_value",
        "baseline_value", "day_value", "ratio", "contribution",
    ]].copy()
    display.columns = [
        "", "Dimension", "Value", "Baseline", "This bucket", "Ratio", "Contribution",
    ]

    st.dataframe(
        display,
        hide_index=True,
        key=f"{key_prefix}_day_movers",
        column_config={
            "Ratio": st.column_config.NumberColumn(format="%.3fx"),
            "Contribution": st.column_config.NumberColumn(
                format="%.4f", help="Share of the bucket's total change, in metric units"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Drill-down 2 — one dimension, from a click on the dispersion bar chart
# ---------------------------------------------------------------------------

def render_dimension_panel(client, verdict: dict, dimension_name: str, grain, key_prefix: str):
    metric = verdict["metric"]
    window_values = _parse_window(verdict["window"], grain)

    st.markdown(f"#### 🧩 Inside `{dimension_name}`")

    with st.spinner(f"Breaking down {dimension_name}..."):
        try:
            values = cached_dimension_values(client, metric, tuple(window_values), dimension_name, grain)
        except Exception as e:
            st.warning(f"Dimension drill-down failed: {safe_error(e)}")
            return

    if values.empty:
        st.info("No values returned for this dimension.")
        return

    tab_table, tab_waterfall, tab_heatmap = st.tabs(
        ["📋 All values", "💧 Contribution waterfall", "🔥 Onset heatmap"]
    )

    # --- every value in the dimension -------------------------------------
    with tab_table:
        st.caption(
            "Sorted by contribution — how much of the metric's total change this value "
            "accounts for. Select a row to chart that segment against the platform. "
            "Note: these ratios are volume-weighted (ratio-of-sums), while the verdict "
            "above uses average-of-bucket-ratios, so the two can differ slightly."
        )
        display = values.copy()
        display["direction"] = ["🔻" if r < 1 else "🔺" for r in display["ratio"].fillna(1)]
        display["volume_share_pct"] = display["volume_share"].astype(float) * 100
        display["contribution_share_pct"] = display["contribution_share"].astype(float) * 100
        table_df = display[[
            "direction", "dimension_value", "baseline_value", "window_value",
            "ratio", "volume_share_pct", "contribution_share_pct",
        ]].copy()
        table_df.columns = [
            "", "Value", "Baseline", "During incident", "Ratio", "Volume share", "Share of change",
        ]

        event = st.dataframe(
            table_df,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"{key_prefix}_values",
            column_config={
                "Ratio": st.column_config.NumberColumn(format="%.3fx"),
                "Volume share": st.column_config.NumberColumn(format="%.1f%%"),
                "Share of change": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Contribution to the total move. Values above ~50% mean this one "
                         "segment explains most of the incident.",
                ),
            },
        )

        rows = _selected_rows(event)
        if rows:
            picked_value = values.iloc[rows[0]]["dimension_value"]
            render_segment_timeseries(
                client, metric, dimension_name, picked_value, window_values, grain,
                key_prefix=f"{key_prefix}_seg",
            )
        else:
            st.caption("👆 Select a row to see that segment's series over time.")

    # --- waterfall --------------------------------------------------------
    with tab_waterfall:
        st.caption(
            "Top movers plus an 'everything else' bucket. The bars sum to the metric's "
            "total change over the incident window."
        )
        top = values.head(8).copy()
        rest = values.iloc[8:]
        rows_wf = [
            {"label": str(r["dimension_value"]), "value": float(r["contribution"] or 0)}
            for _, r in top.iterrows()
        ]
        if not rest.empty:
            rows_wf.append(
                {"label": f"Everything else ({len(rest)})", "value": float(rest["contribution"].fillna(0).sum())}
            )

        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(rows_wf) + ["total"],
            x=[r["label"] for r in rows_wf] + ["Total change"],
            y=[r["value"] for r in rows_wf] + [0],
            connector={"line": {"color": "#adb5bd"}},
            decreasing={"marker": {"color": "#e63946"}},
            increasing={"marker": {"color": "#2a9d8f"}},
            totals={"marker": {"color": "#457b9d"}},
        ))
        wf.update_layout(
            height=420, margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title=f"Change in {chq.METRIC_LABELS[metric]}",
            xaxis_tickangle=-30, showlegend=False,
        )
        st.plotly_chart(wf, width="stretch", key=f"{key_prefix}_waterfall")

    # --- heatmap ----------------------------------------------------------
    with tab_heatmap:
        st.caption(
            "Ratio against each value's own baseline, bucket by bucket. Shows whether the "
            "problem hit one segment on one bucket or bled across segments over time. "
            "Top 15 values by volume only."
        )
        try:
            hm = cached_onset_heatmap(client, metric, tuple(window_values), dimension_name, grain)
        except Exception as e:
            st.warning(f"Heatmap query failed: {safe_error(e)}")
            return

        if hm.empty:
            st.info("No data for the heatmap.")
            return

        pivot = hm.pivot(index="dimension_value", columns="t", values="ratio")
        hm_fig = px.imshow(
            pivot,
            color_continuous_scale=["#e63946", "#f8f9fa", "#2a9d8f"],
            color_continuous_midpoint=1.0,
            aspect="auto",
            labels=dict(color="Ratio vs baseline"),
        )
        hm_fig.update_layout(height=460, margin=dict(l=10, r=10, t=20, b=10),
                             xaxis_title="", yaxis_title="")
        st.plotly_chart(hm_fig, width="stretch", key=f"{key_prefix}_heatmap")
        st.caption(
            "Incident window: "
            + (f"{window_values[0]} → {window_values[-1]}" if len(window_values) > 1 else str(window_values[0]))
            + ". Red = below that value's own baseline, green = above."
        )


# ---------------------------------------------------------------------------
# Drill-down 3 — one segment's series vs the platform
# ---------------------------------------------------------------------------

def render_segment_timeseries(client, metric: str, dimension_name: str, dimension_value: str,
                               window_values, grain, key_prefix: str):
    st.markdown(f"**`{dimension_name} = {dimension_value}` vs platform**")
    try:
        ts = cached_segment_timeseries(client, metric, dimension_name, str(dimension_value), grain)
    except Exception as e:
        st.warning(f"Segment series failed: {safe_error(e)}")
        return

    if ts.empty:
        st.info("No series returned for this segment.")
        return

    # index both to 100 at their pre-incident mean so they share one axis
    mask = ~ts["t"].isin(window_values)
    seg_base = ts.loc[mask, "segment_value"].mean()
    tot_base = ts.loc[mask, "total_value"].mean()

    plot = ts.copy()
    plot["Segment"] = plot["segment_value"] / seg_base * 100 if seg_base else plot["segment_value"]
    plot["Platform"] = plot["total_value"] / tot_base * 100 if tot_base else plot["total_value"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot["t"], y=plot["Platform"], name="Platform",
                             mode="lines", line=dict(color="#adb5bd", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=plot["t"], y=plot["Segment"], name=str(dimension_value),
                             mode="lines+markers", line=dict(color="#e63946", width=2)))
    if window_values:
        fig.add_vrect(x0=window_values[0], x1=window_values[-1],
                      fillcolor="#e63946", opacity=0.10, line_width=0)
    fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="#6c757d")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Indexed to 100 = pre-incident average",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_ts")
    st.caption(
        "Both lines indexed to their own pre-incident average. If the red line dives while "
        "the grey one holds, the segment is the story. If they move together, it isn't."
    )


# ---------------------------------------------------------------------------
# Drill-down 4 — revenue decomposition (volume vs monetisation)
# ---------------------------------------------------------------------------

def render_revenue_decomposition(client, window_values, grain, key_prefix: str):
    st.markdown("#### ⚖️ What drove the revenue move?")
    st.caption(
        "revenue = requests × fill rate × render rate × eCPM/1000. That identity is exact, "
        "so in log space the four drivers add up to the total move. This is the "
        "volume-versus-monetisation question, answered directly."
    )
    try:
        dec = cached_revenue_decomposition(client, tuple(window_values), grain)
    except Exception as e:
        st.warning(f"Decomposition failed: {safe_error(e)}")
        return

    if dec.empty:
        st.info("No decomposition data.")
        return

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(dec) + ["total"],
        x=list(dec["factor"]) + ["Total revenue move"],
        y=list(dec["log_delta"].astype(float)) + [0],
        text=[f"{v:+.1f}%" for v in dec["pct_change"]] + [""],
        textposition="outside",
        connector={"line": {"color": "#adb5bd"}},
        decreasing={"marker": {"color": "#e63946"}},
        increasing={"marker": {"color": "#2a9d8f"}},
        totals={"marker": {"color": "#457b9d"}},
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="Log change (additive)", showlegend=False)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_decomp")

    residual = dec.attrs.get("residual")
    if residual is not None and pd.notna(residual) and abs(residual) > 1e-6:
        st.warning(
            f"Identity check failed: the four drivers are off the total by {residual:.6f} "
            "in log space. A denominator was zero somewhere in this window — treat the "
            "split as unreliable."
        )

    ranked = dec.dropna(subset=["log_delta"])
    if ranked.empty:
        st.caption("Decomposition returned no usable values — check for zero denominators in the window.")
        return
    biggest = ranked.loc[ranked["log_delta"].astype(float).abs().idxmax()]
    st.caption(
        f"Largest single driver: **{biggest['factor']}** at {biggest['pct_change']:+.1f}%. "
        "Bars are log changes so they sum exactly; the labels are the equivalent percentage moves."
    )