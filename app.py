import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Every chart in the app (here and in drilldowns.py, imported below and run
# in this same process) inherits this — one line instead of restyling each
# figure individually. The one existing explicit background override (the
# geo map's transparent geos) is compatible with a dark template, not
# fighting it.
pio.templates.default = "plotly_dark"

import annotations as ann
import drilldowns as dd
import granularity as gr
import langfuse_tracing as lft
from llm_stub import sanitize_env, list_models, chat_models, resolve_model, langfuse_status
from clickhouse_queries import (
    get_client,
    step1_trigger_scan,
    find_incident_windows,
    build_verdict,
    METRIC_LABELS,
)

load_dotenv()
sanitize_env()   # strip stray quotes from .env values before any SDK reads them

st.set_page_config(page_title="Heimdall for InMobi", layout="wide", page_icon="👁️")

# ---------------------------------------------------------------------------
# Button text-contrast fix — the one real gap left after moving to
# config.toml's native theming (see .streamlit/config.toml for the full
# rationale). There is no config.toml key for "primary button text color"
# independent of its background, so a bright yellow primary button gets
# whatever text color Streamlit's component library defaults to — which
# was unreadable white-on-yellow.
#
# Targeted via Streamlit's documented `.st-key-<name>` classes (generated
# automatically from each widget's `key=`) rather than generic selectors
# like `.stButton > button` — Streamlit has a confirmed, open bug (GitHub
# issue #10384) where broadly-targeted injected <style> can be present in
# the DOM but silently fail to apply; the key-based classes are the
# pattern Streamlit's own current guidance recommends specifically because
# it's more reliable in practice.
st.markdown(
    """
    <style>
    .st-key-scan_button button {
        color: #000000 !important;
    }
    .st-key-scan_button button p {
        color: inherit !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session lifecycle — explicit, not incidental.
#
# A genuinely new browser tab already gets an empty st.session_state from
# Streamlit for free (every new WebSocket connection is a new session), so
# stale scan results from someone else's tab can never leak in on their own.
# What this block adds on top: (1) it is explicit and documented rather than
# relying on that framework behavior silently, and (2) it backs the "Start
# new session" button below, which gives a deliberate reset inside the SAME
# tab — for when a hard refresh or process restart isn't what you want, just
# a clean slate before the next scan.
# ---------------------------------------------------------------------------
if "_session_initialized" not in st.session_state:
    st.session_state["scanned"] = False
    lft.get_session_id()  # mint one now so it's stable for the whole session
    st.session_state["_session_initialized"] = True

METRIC_COLORS = {"revenue": "#e63946", "fill_rate": "#f4a261", "ecpm": "#2a9d8f"}
METRIC_ORDER = ("revenue", "fill_rate", "ecpm")
ROW_OF_METRIC = {"revenue": 1, "fill_rate": 2, "ecpm": 3}


# ---------------------------------------------------------------------------
# Connection — cached as a resource so drill-down reruns reuse one connection
# instead of opening a new one on every click.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_client():
    return get_client()


# ---------------------------------------------------------------------------
# Sidebar — connection + scan controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Scan settings")

# Granularity — Daily / Hourly. Only the label is shown; the underlying table
# name and time column both come from the environment (granularity.py) and
# are never rendered.
GRAINS = gr.load_grains()
grain_label = st.sidebar.radio(
    "Table", [GRAINS["daily"].label, GRAINS["hourly"].label], horizontal=True,
)
grain_key = "daily" if grain_label == GRAINS["daily"].label else "hourly"
grain = GRAINS[grain_key]

# Switching granularity invalidates any scan result from the OTHER grain —
# without this, flipping the toggle would keep showing daily incidents next
# to an "Hourly" label until Scan is clicked again.
if st.session_state.get("scanned_grain") not in (None, grain_key):
    st.session_state["scanned"] = False
    st.sidebar.caption("↻ Table changed — click Scan to load it.")

if grain_key == "hourly":
    st.sidebar.caption(f"⚠️ {grain.min_history_hint}.")

z_threshold = st.sidebar.slider(
    "Anomaly z-score threshold", 1.5, 4.0, 2.0, 0.1,
    help="How many standard deviations from the period average a point must be to flag. "
         "Lower = more sensitive, more false positives. Higher = fewer, larger-only incidents.",
)
min_contribution_share = st.sidebar.slider(
    "Minimum contribution share to call a culprit", 0.1, 0.9, 0.5, 0.05,
    help="A culprit is named when one (dimension, value) pair explains at least this share "
         "of the total observed change — e.g. 0.5 means 'this one segment accounts for at "
         "least half the move.' Below this, the incident is treated as platform-wide. "
         "Replaces the old dispersion-ratio threshold, which could rank a mild 3-way mix "
         "shift above a genuinely broken single value in a larger dimension.",
)
min_volume_share = st.sidebar.slider(
    "Minimum segment volume share", 0.0, 0.10, 0.01, 0.005,
    help="A segment must carry at least this fraction of its dimension's baseline volume "
         "to be considered — guards against a near-empty segment producing a wild ratio "
         "and a misleadingly large contribution purely from noise.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart options")
show_band = st.sidebar.checkbox(
    "Show detection band on trend", value=True,
    help="Shades the normal range the scan uses — anything outside it is what triggers a flag. "
         "Now computed per seasonal bucket (same weekday for Daily, same weekday+hour for "
         "Hourly), not a single flat range for the whole period.",
)
band_style = "Percentile (P10–P90)"
if show_band:
    band_style = st.sidebar.radio(
        "Band style", ["Percentile (P10–P90)", "Parametric (mean ± z·σ)"],
        help="Percentile bands make no assumption about the shape of the distribution — "
             "recommended when a metric doesn't look bell-curve-shaped. Parametric is the "
             "classic mean ± z·σ corridor, now also computed per seasonal bucket.",
    )
show_volume = st.sidebar.checkbox(
    "Overlay request volume", value=False,
    help="Adds a secondary axis showing raw request volume, so you can see at a glance whether "
         "a metric move was a traffic change or a monetisation change.",
)

st.sidebar.markdown("---")

scan_clicked = st.sidebar.button(
    "🔍 Scan for Anomalies", type="primary", key="scan_button",
    help="Runs the full trigger scan + culprit ranking against the selected table.",
)

if st.sidebar.button(
    "♻️ Clear cached queries", key="clear_cache_button",
    help="Forces every drill-down and LLM report to re-run instead of serving a cached result. "
         "Also the only way to get a fresh Langfuse trace for something already on screen.",
):
    st.cache_data.clear()
    st.toast("Drill-down cache cleared.")

if st.sidebar.button(
    "🆕 Start new session", key="new_session_button",
    help="Resets scan results and mints a new Langfuse session id, without needing a hard "
         "browser refresh or process restart. Annotations you've saved are unaffected — "
         "those are stored durably, not tied to a session.",
):
    st.session_state["scanned"] = False
    st.session_state.pop("verdicts", None)
    st.session_state.pop("trigger_df", None)
    new_id = lft.new_session()
    st.toast(f"New session: {new_id}")
    st.rerun()

with st.sidebar.expander("🤖 LLM model", expanded=False):
    st.caption(
        "Groq disables models per project, so the app asks your account what it will "
        "serve and picks the smallest chat model. Override here if you want a specific one."
    )
    if st.button("Refresh model list", key="refresh_models_button", width="stretch"):
        ok, result = list_models()
        if ok:
            st.session_state["groq_models"] = chat_models(result)
            st.toast(f"{len(st.session_state['groq_models'])} chat model(s) enabled")
        else:
            st.session_state["groq_models"] = []
            st.error(result)

    available = st.session_state.get("groq_models")
    if available:
        chosen = st.selectbox("Model", ["Auto (smallest enabled)"] + available)
        st.session_state["groq_model"] = None if chosen.startswith("Auto") else chosen
    elif available == []:
        st.caption("No chat models enabled. Enable one under console.groq.com project limits.")
    else:
        st.caption("Not checked yet — the app will auto-resolve on the first report.")

    active = st.session_state.get("groq_model") or resolve_model()
    st.caption(f"Active: `{active}`" if active else "Active: placeholder (no model available)")

with st.sidebar.expander("📡 Langfuse tracing", expanded=False):
    recheck = st.button("Re-check connection", key="recheck_langfuse_button", width="stretch")
    status = langfuse_status(force=recheck)
    if status["active"]:
        st.success(f"Tracing active → {status['base_url']}")
    else:
        st.warning(f"Not tracing: {status['reason']}")
        st.caption(status["debug_hint"])
        st.caption(
            "If you just added or changed Langfuse keys in .env, this will not pick them "
            "up on a page rerun — restart `streamlit run app.py`. The Langfuse client is a "
            "singleton created once per process and cached from that point on."
        )
    st.caption(
        "Either LANGFUSE_BASE_URL or the older LANGFUSE_HOST works (confirmed against the "
        "installed SDK source); LANGFUSE_BASE_URL takes priority if both are set. "
        "Every trace in this browser tab shares one session — see it grouped under "
        "Sessions in the Langfuse UI."
    )

    stats = lft.span_stats()
    st.markdown("---")
    st.caption(
        f"**This session:** {stats['emitted']} span(s) opened, {stats['skipped']} skipped "
        "(tracing off, or setup failed)."
    )
    st.caption(
        "If this shows 0 emitted after clicking around, nothing new actually ran — every "
        "drill-down and the scan itself only trace on a genuine cache miss. Click "
        "**♻️ Clear cached queries** above, then interact again (new chart points, new "
        "dimension bars — not ones already opened in this tab within the last 10 minutes) "
        "to force fresh, traced calls."
    )

st.sidebar.markdown(
    "<div style='margin-top:2rem; opacity:0.55; font-size:0.75rem;'>"
    "👁️ Heimdall for InMobi<br>Built by Team Data Heimdall</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Masthead
#
# The gradient bar below is deliberate, not decorative: its three colours are
# the exact METRIC_COLORS used on every chart in this app (revenue, fill
# rate, eCPM). It reads as the Bifrost — Heimdall's bridge — while actually
# encoding what this tool watches, so the mythology and the function are the
# same object rather than a logo bolted on top of it.
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* Space Grotesk stands in for ClickHouse's own display face, Basier
       Square — that one is a commercial/licensed font with no public CDN
       source, so this is a close-in-spirit substitute, not the real thing. */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap');

    .heimdall-eyebrow {{
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #FAFF69; margin-bottom: 0.15rem;
    }}
    .heimdall-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem; font-weight: 700; letter-spacing: -0.01em;
        margin: 0; line-height: 1.15; color: #F2F2F0;
    }}
    .heimdall-tagline {{
        font-size: 1rem; font-style: italic; color: #9AA0A6; margin-top: 0.2rem;
    }}
    .heimdall-bridge {{
        height: 4px; border-radius: 2px; margin: 0.9rem 0 1.1rem 0;
        background: linear-gradient(
            90deg,
            {METRIC_COLORS['revenue']} 0%,
            {METRIC_COLORS['fill_rate']} 50%,
            {METRIC_COLORS['ecpm']} 100%
        );
    }}
    </style>
    <div class="heimdall-eyebrow">Team Data Heimdall</div>
    <p class="heimdall-title">👁️ Heimdall for InMobi</p>
    <p class="heimdall-tagline">Sees everything. Names the culprit.</p>
    <div class="heimdall-bridge"></div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Platform-wide trigger scan → dispersion-ranked culprit search across every dimension → "
    "named culprit value. All detection math runs in SQL; this page renders the results. "
    "**Charts are clickable** — click a trend point or a dispersion bar to drill in."
)

# ---------------------------------------------------------------------------
# Run the pipeline on click
# ---------------------------------------------------------------------------
if scan_clicked:
    with st.spinner("Connecting to ClickHouse and scanning..."):
        try:
            client = get_cached_client()
            with lft.traced_root(
                "anomaly-scan",
                input={"grain": grain.key, "z_threshold": z_threshold,
                      "min_contribution_share": min_contribution_share,
                      "min_volume_share": min_volume_share},
            ) as scan_span:
                with lft.traced("scan.trigger_scan", input={"grain": grain.key}) as span:
                    trigger_df = step1_trigger_scan(client, grain)
                    span.update(output=lft.summarize_df(trigger_df))

                incidents = find_incident_windows(trigger_df, grain, z_threshold=z_threshold)

                verdicts = []
                for metric, windows in incidents.items():
                    for window_values in windows:
                        with lft.traced(
                            "scan.build_verdict",
                            input={"metric": metric, "window": [str(w) for w in window_values]},
                        ) as span:
                            v = build_verdict(
                                client, metric, window_values, grain,
                                min_contribution_share=min_contribution_share,
                                min_volume_share=min_volume_share,
                            )
                            if v:
                                span.update(output={
                                    "has_culprit": v["has_culprit"],
                                    "culprit_dimension": v.get("culprit_dimension"),
                                })
                                # captured while the span is still open — these
                                # ids are only readable inside the `with` block
                                v.update(lft.current_ids())
                                verdicts.append(v)

                scan_span.update(output={"incidents_found": len(verdicts)})
                scan_ids = lft.current_ids()  # captured while scan_span is still open

            st.session_state["trigger_df"] = trigger_df
            st.session_state["verdicts"] = verdicts
            st.session_state["z_threshold"] = z_threshold
            st.session_state["scanned_grain"] = grain_key
            st.session_state["scanned"] = True
            st.session_state["scan_ids"] = scan_ids
        except Exception as e:
            st.error(f"Scan failed: {dd.safe_error(e)}")
            st.stop()

if not st.session_state.get("scanned"):
    st.info("👈 Click **Scan for Anomalies** in the sidebar to run the pipeline.")
    st.stop()

trigger_df = st.session_state["trigger_df"]
verdicts = st.session_state["verdicts"]
z_threshold = st.session_state["z_threshold"]
client = get_cached_client()

# in-memory only — a fact about this scan's results, not a durable judgement
correlated = ann.correlate(verdicts)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
n_total = len(verdicts)
n_by_metric = {m: sum(1 for v in verdicts if v["metric"] == m) for m in METRIC_ORDER}
n_with_culprit = sum(1 for v in verdicts if v["has_culprit"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total anomalies", n_total,
         help="Every window flagged by the z-score trigger scan, across all three metrics.")
k2.metric("💰 Revenue", n_by_metric["revenue"], help="Revenue incidents in this scan.")
k3.metric("📶 Fill rate", n_by_metric["fill_rate"], help="Fill rate incidents in this scan.")
k4.metric("💵 eCPM", n_by_metric["ecpm"], help="eCPM incidents in this scan.")
k5.metric("🎯 With culprit found", f"{n_with_culprit} / {n_total}" if n_total else "0 / 0",
         help="Incidents where one dimension's value dominated the move, versus a platform-wide "
              "move with no single segment to blame.")

# ---------------------------------------------------------------------------
# Metric filter (radio, not a donut click — Streamlit selection events do not
# cover pie traces reliably, so the donut stays read-only and the radio drives
# the filter for the rest of the page)
# ---------------------------------------------------------------------------
present = [m for m in METRIC_ORDER if n_by_metric[m] > 0]

if n_total > 0:
    donut_col, filter_col = st.columns([1, 2])
    with donut_col:
        donut_df = pd.DataFrame(
            [{"Metric": METRIC_LABELS[m], "Count": n_by_metric[m]} for m in present]
        )
        if not donut_df.empty:
            donut_fig = px.pie(
                donut_df, names="Metric", values="Count", hole=0.55,
                color="Metric",
                color_discrete_map={
                    "Revenue": METRIC_COLORS["revenue"],
                    "Fill Rate": METRIC_COLORS["fill_rate"],
                    "eCPM": METRIC_COLORS["ecpm"],
                },
            )
            donut_fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                                     legend=dict(orientation="h", y=-0.15))
            donut_fig.update_traces(textinfo="value+label")
            st.plotly_chart(donut_fig, width="stretch", key="donut")
    with filter_col:
        st.markdown("**Filter incidents by metric**")
        metric_filter = st.radio(
            "Metric filter",
            ["All"] + [METRIC_LABELS[m] for m in present],
            horizontal=True, label_visibility="collapsed",
        )
else:
    metric_filter = "All"

visible = (
    verdicts if metric_filter == "All"
    else [v for v in verdicts if v["metric_label"] == metric_filter]
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Trend charts — clickable, with detection band and optional volume overlay
# ---------------------------------------------------------------------------
st.subheader(f"📈 Anomaly Trend — Revenue, Fill Rate & eCPM ({grain.label})")

scan_ids = st.session_state.get("scan_ids") or {}
if scan_ids.get("trace_url"):
    st.caption(
        f"🔗 [View this scan's trace in Langfuse]({scan_ids['trace_url']}) · "
        f"`trace_id={scan_ids.get('trace_id')}` — every incident below nests under this same trace."
    )

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=("Revenue", "Fill Rate", "eCPM"),
    vertical_spacing=0.08,
    specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
)

# maps plotly curve_number -> metric, so a click can be attributed correctly
curve_metric = []

for i, metric in enumerate(METRIC_ORDER, start=1):
    # Detection band — now drawn per-row rather than as one flat ribbon,
    # because the baseline itself is seasonal: a Tuesday's normal range is
    # no longer the same number as a Saturday's. Two styles available:
    #   Percentile: literal P10/P90 of that row's own seasonal bucket —
    #     makes no distributional assumption, robust to skew.
    #   Parametric: mean ± z·σ of that row's own seasonal bucket — matches
    #     exactly what the trigger scan's z-score is computed against.
    if show_band and f"{metric}_mean" in trigger_df.columns:
        if band_style.startswith("Percentile"):
            lo_series = trigger_df[f"{metric}_p10"]
            hi_series = trigger_df[f"{metric}_p90"]
        else:
            lo_series = trigger_df[f"{metric}_mean"] - z_threshold * trigger_df[f"{metric}_std"]
            hi_series = trigger_df[f"{metric}_mean"] + z_threshold * trigger_df[f"{metric}_std"]

        fig.add_trace(
            go.Scatter(x=trigger_df["t"], y=lo_series, mode="lines",
                       line=dict(width=0), hoverinfo="skip", showlegend=False),
            row=i, col=1, secondary_y=False,
        )
        curve_metric.append(None)
        fig.add_trace(
            go.Scatter(x=trigger_df["t"], y=hi_series, mode="lines",
                       line=dict(width=0), fill="tonexty", fillcolor="rgba(69,123,157,0.10)",
                       hoverinfo="skip", showlegend=False),
            row=i, col=1, secondary_y=False,
        )
        curve_metric.append(None)

    is_anom = trigger_df[f"{metric}_z"].abs() > z_threshold
    fig.add_trace(
        go.Scatter(
            x=trigger_df["t"], y=trigger_df[metric], mode="lines+markers",
            name=METRIC_LABELS[metric],
            marker=dict(
                color=["#e63946" if a else "#457b9d" for a in is_anom],
                size=[10 if a else 5 for a in is_anom],
            ),
            line=dict(color="#457b9d"),
            customdata=np.stack([
                trigger_df[f"{metric}_z"], trigger_df[f"{metric}_p90"],
                trigger_df[f"{metric}_p95"], trigger_df[f"{metric}_p99"],
                trigger_df["bucket_n"],
            ], axis=-1),
            hovertemplate=(
                "%{x}<br>%{y:.4f}<br>z = %{customdata[0]:.2f}<br>"
                "P90/P95/P99 (this bucket): %{customdata[1]:.2f} / %{customdata[2]:.2f} / "
                "%{customdata[3]:.2f}<br>bucket sample size: %{customdata[4]}<extra></extra>"
            ),
        ),
        row=i, col=1, secondary_y=False,
    )
    curve_metric.append(metric)

    if show_volume and "requests" in trigger_df.columns:
        fig.add_trace(
            go.Scatter(x=trigger_df["t"], y=trigger_df["requests"], mode="lines",
                       name="Requests", line=dict(color="#adb5bd", width=1, dash="dot"),
                       hovertemplate="%{x}<br>requests %{y:,.0f}<extra></extra>",
                       showlegend=(i == 1)),
            row=i, col=1, secondary_y=True,
        )
        curve_metric.append(None)

for v in visible:
    fig.add_vrect(
        x0=v["window"][0], x1=v["window"][-1],
        fillcolor=METRIC_COLORS[v["metric"]], opacity=0.12, line_width=0,
        row=ROW_OF_METRIC[v["metric"]], col=1,
    )

fig.update_layout(height=680, showlegend=show_volume, margin=dict(t=40, b=20),
                  clickmode="event+select")
fig.update_yaxes(showgrid=False, secondary_y=True)

trend_event = st.plotly_chart(
    fig, width="stretch", key="trend_chart", on_select="rerun"
)
st.caption(
    "Red markers = flagged points (|z| above threshold). Shaded band = the grouped incident "
    "window. The pale corridor is the **seasonal** normal range — computed per weekday (Daily) "
    "or per weekday+hour (Hourly) where there's enough history, so a normal Saturday no longer "
    "gets compared against a Tuesday's average. **Click any point to inspect that bucket.**"
)

if "baseline_tier" in trigger_df.columns:
    tier_counts = trigger_df["baseline_tier"].value_counts()
    n_seasonal = int(tier_counts.get("seasonal", 0))
    n_daytype = int(tier_counts.get("day_type", 0))
    n_flat = int(tier_counts.get("flat", 0))
    total_buckets = len(trigger_df)

    if n_daytype or n_flat:
        parts = []
        if n_daytype:
            parts.append(f"{n_daytype} bucket(s) fell back to a coarser weekday-vs-weekend "
                         "baseline (not enough history yet for the full weekday split)")
        if n_flat:
            parts.append(f"{n_flat} bucket(s) fell all the way back to a flat, whole-period "
                         "baseline (not enough history for weekday-vs-weekend either) — these "
                         "are the least reliable rows on the chart, since a flat baseline is "
                         "still mixed with real seasonality it can't separate out")
        st.caption(
            f"📊 Baseline used: {n_seasonal}/{total_buckets} bucket(s) at full seasonal precision. "
            + "; ".join(parts) + "."
        )
    min_bucket_n = int(trigger_df["bucket_n"].min())
    if min_bucket_n < 20:
        st.caption(
            f"⚠️ The smallest fully-seasonal bucket has only **{min_bucket_n}** historical points "
            "behind it. With this few samples, P95/P99 are close to just the bucket's max — "
            "informative for a quick look, not a number to cite precisely. More history makes "
            "every tier above more reliable, especially on Hourly (up to 168 buckets vs Daily's 7)."
        )

points = dd._selected_points(trend_event)
if points:
    pt = points[0]
    picked_metric = curve_metric[pt.get("curve_number", 0)] if pt.get("curve_number", 0) < len(curve_metric) else None
    if picked_metric is None:
        st.info("That trace isn't drillable — click a coloured metric point instead.")
    else:
        picked_bucket = pd.to_datetime(pt["x"])
        picked_bucket = picked_bucket.to_pydatetime() if grain.is_datetime else picked_bucket.date()
        with st.container(border=True):
            dd.render_day_panel(client, picked_metric, picked_bucket, grain, key_prefix="trend")
else:
    st.caption("👆 No bucket selected. Click a point on any of the three charts above.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Summary table — selectable, drives which incident opens below
# ---------------------------------------------------------------------------
st.subheader(f"🧾 Anomalies found: {len(visible)}" + ("" if metric_filter == "All" else f"  ({metric_filter})"))

focus_idx = 0
if not visible:
    st.success("No anomalies above the current z-score threshold. Try lowering the threshold in the sidebar.")
else:
    summary_rows = []
    for v in visible:
        window_label = f"{v['window'][0]} → {v['window'][-1]}" if len(v["window"]) > 1 else v["window"][0]
        fp = ann.fingerprint(grain.key, v)
        a = ann.get(fp)
        summary_rows.append({
            "Metric": v["metric_label"],
            "Window": window_label,
            "Buckets": len(v["window"]),
            "Detection": "🔴 Culprit found" if v["has_culprit"] else "🟡 Platform-wide (no single culprit)",
            "Culprit": f"{v.get('culprit_dimension', '—')} = {v.get('culprit_value', '—')}" if v["has_culprit"] else "—",
            "Ratio vs baseline": f"{v.get('ratio', 1.0):.2f}x" if v["has_culprit"] else "—",
            "Acknowledged": a["status"] if a else "New",
        })
    summary_event = st.dataframe(
        pd.DataFrame(summary_rows), width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="summary_table",
        column_config={
            "Detection": st.column_config.TextColumn(
                help="What the trigger scan and culprit ranking found."
            ),
            "Acknowledged": st.column_config.TextColumn(
                help="Your team's own status for this incident — set it in the incident detail below."
            ),
        },
    )
    sel_rows = dd._selected_rows(summary_event)
    if sel_rows:
        focus_idx = sel_rows[0]
    st.caption("👆 Select a row to open that incident below.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-incident detail
# ---------------------------------------------------------------------------
st.subheader("🔬 Incident Detail — Culprit Ranking & Diagnosis")


def direction_word(ratio: float) -> str:
    return "dropped" if ratio < 1 else "spiked"


for idx, v in enumerate(visible):
    window_label = f"{v['window'][0]} → {v['window'][-1]}" if len(v["window"]) > 1 else v["window"][0]
    header_emoji = "🔴" if v["has_culprit"] else "🟡"

    fp = ann.fingerprint(grain.key, v)
    existing_annotation = ann.get(fp)
    status_emoji = {
        "New": "", "Investigating": "🔎 ", "Resolved": "✅ ", "False positive": "🚫 ",
    }.get(existing_annotation["status"] if existing_annotation else "New", "")

    with st.expander(
        f"{header_emoji} {status_emoji}{v['metric_label']} anomaly — {window_label}",
        expanded=(idx == focus_idx),
    ):
        # --- cross-incident correlation callout ---------------------------
        v_key = (v.get("culprit_dimension"), v.get("culprit_value"))
        if v_key in correlated:
            siblings = [visible[j]["metric_label"] for j in correlated[v_key] if visible[j] is not v]
            if siblings:
                st.warning(
                    f"🔗 **{v['culprit_dimension']} = {v['culprit_value']}** also shows up as the "
                    f"culprit in this scan's **{', '.join(siblings)}** incident(s) — likely one "
                    "underlying cause, not a coincidence."
                )

        left, right = st.columns([1.3, 1])

        ranking_df = pd.DataFrame(v["ranking"])
        ranking_df["status"] = [
            "Culprit" if (v["has_culprit"] and i == 0) else "Ruled out"
            for i in range(len(ranking_df))
        ]

        with left:
            st.markdown("**Culprit ranking — contribution share across all dimensions**")
            st.caption(
                "Bigger bar = more of the total observed change is explained by that "
                "dimension's single best value. Fixed from an earlier version that ranked "
                "by dispersion (how spread-out a dimension's values were) — that approach "
                "favored low-cardinality dimensions with a mild mix-shift over a "
                "high-cardinality dimension with one genuinely broken value. "
                "**Click a bar to open that dimension.**"
            )
            chart_df = ranking_df.sort_values("top_contribution_share", ascending=True).copy()
            chart_df["display_label"] = chart_df["dimension_name"].where(
                chart_df["dimension_name"] != "country", "country 🗺️"
            )
            bar_fig = px.bar(
                chart_df,
                x="top_contribution_share", y="display_label", orientation="h",
                color="status",
                color_discrete_map={"Culprit": "#e63946", "Ruled out": "#2a9d8f"},
                text="top_contribution_share",
                custom_data=["top_value", "top_ratio", "dispersion"],
            )
            bar_fig.update_traces(
                texttemplate="%{text:.3f}", textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Top value: %{customdata[0]}<br>"
                    "Contribution share: %{x:.3f}<br>"
                    "That value's own ratio: %{customdata[1]:.3f}x<br>"
                    "Dispersion (legacy stat, no longer the decision criterion): %{customdata[2]:.4f}"
                    "<extra></extra>"
                ),
            )
            bar_fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Contribution share (higher = more suspicious)", yaxis_title="",
                clickmode="event+select",
            )
            bar_event = st.plotly_chart(
                bar_fig, width="stretch", key=f"bar_{idx}", on_select="rerun"
            )

        with right:
            if v["has_culprit"]:
                st.markdown(f"**Culprit: `{v['culprit_dimension']} = {v['culprit_value']}`**")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=v["window_value"],
                    delta={"reference": v["baseline_value"], "relative": True, "valueformat": ".1%"},
                    gauge={
                        "axis": {"range": [0, max(v["baseline_value"], v["window_value"]) * 1.3 or 1]},
                        "bar": {"color": "#e63946" if v["ratio"] < 1 else "#2a9d8f"},
                    },
                    title={"text": f"{v['metric_label']} — incident vs baseline"},
                ))
                gauge_fig.update_layout(height=280, margin=dict(l=10, r=10, t=60, b=10))
                # go.Indicator has no hoverinfo/hovertemplate property at all — verified
                # against the installed plotly, not assumed. The caption right below
                # already states baseline/incident/ratio explicitly, so nothing lost.
                st.plotly_chart(gauge_fig, width="stretch", key=f"gauge_{idx}")
                st.caption(
                    f"Baseline: **{v['baseline_value']}** → During incident: **{v['window_value']}** "
                    f"({v['ratio']:.2f}x, {direction_word(v['ratio'])} {abs(1 - v['ratio']) * 100:.1f}%)"
                )
            else:
                st.markdown("**No single culprit dimension** 🟡")
                st.caption(v.get("note", ""))

        # --- dimension drill-down, driven by the bar click ------------------
        window_values = dd._parse_window(v["window"], grain)
        bar_points = dd._selected_points(bar_event)
        default_dim = v.get("culprit_dimension") or (ranking_df["dimension_name"].iloc[0] if not ranking_df.empty else None)
        has_country = "country" in ranking_df["dimension_name"].values

        jump_col, _ = st.columns([1, 3])
        with jump_col:
            if has_country and st.button(
                "🗺️ Jump to country map", key=f"jump_map_{idx}",
                help="Opens the country dimension directly, regardless of which bar is "
                     "currently selected above — the map only ever appears for `country`, "
                     "and is easy to miss if the culprit was a different dimension.",
            ):
                st.session_state[f"picked_dim_{idx}"] = "country"

        if bar_points:
            clicked = str(bar_points[0].get("y", "")).replace(" 🗺️", "").strip()
            st.session_state[f"picked_dim_{idx}"] = clicked or default_dim
        picked_dim = st.session_state.get(f"picked_dim_{idx}", default_dim)

        if picked_dim:
            with st.container(border=True):
                dd.render_dimension_panel(client, v, str(picked_dim), grain, key_prefix=f"inc{idx}")

        # --- revenue-only: volume vs monetisation decomposition -------------
        if v["metric"] == "revenue":
            with st.container(border=True):
                dd.render_revenue_decomposition(client, window_values, grain, key_prefix=f"inc{idx}")

        st.markdown("**Ruled out dimensions**")
        ruled_out = v.get("ruled_out", [])
        st.write(" ".join(f"`{d}` ✅" for d in ruled_out) if ruled_out else "—")

        # --- acknowledge / annotate ----------------------------------------
        st.markdown("**📝 Acknowledge this incident**")
        st.caption(
            "Saved durably (SQLite), separate from this session — it will still be here "
            "next time this exact incident (same window, same culprit) is re-scanned."
        )
        ack_col1, ack_col2 = st.columns([1, 2])
        with ack_col1:
            current_status = existing_annotation["status"] if existing_annotation else "New"
            new_status = st.selectbox(
                "Status", ann.STATUSES,
                index=ann.STATUSES.index(current_status) if current_status in ann.STATUSES else 0,
                key=f"status_{idx}",
                help="New = not yet looked at. Investigating = someone's on it. Resolved = "
                     "confirmed and handled. False positive = feed this back into your "
                     "threshold tuning if it keeps happening.",
            )
        with ack_col2:
            new_note = st.text_input(
                "Note", value=existing_annotation["note"] if existing_annotation else "",
                key=f"note_{idx}", placeholder="e.g. confirmed with the platform team, rolled back at 14:20",
            )
        if st.button("Save", key=f"save_{idx}"):
            ann.set_annotation(fp, grain.key, v, new_status, new_note)
            st.toast(f"Saved: {new_status}")
            st.rerun()
        if existing_annotation and existing_annotation["updated_count"] > 1:
            st.caption(
                f"Updated {existing_annotation['updated_count']} times — this incident has "
                "recurred or been revisited before."
            )

        # --- trace / span ids ------------------------------------------------
        if v.get("trace_url"):
            st.caption(
                f"🔗 [View this incident's trace in Langfuse]({v['trace_url']}) · "
                f"`trace_id={v.get('trace_id')}` · `span_id={v.get('span_id')}`"
            )
        elif v.get("trace_id") is None:
            st.caption("No Langfuse trace for this incident — tracing was off during this scan.")

        st.markdown("**🧠 Incident report**")
        st.caption(
            "Output of `llm_stub.generate_llm_report()`, built from the trimmed evidence payload "
            "rather than the full verdict. Cached on the evidence, so drill-down clicks do not "
            "re-trigger it."
        )
        dd.render_report(client, v, grain, key_prefix=f"inc{idx}")

st.markdown("---")
st.caption("All detection math runs in SQL · This app formats the results.")