import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

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

st.set_page_config(page_title="InMobi Anomaly Scanner", layout="wide", page_icon="🚨")

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
dispersion_ratio_threshold = st.sidebar.slider(
    "Culprit dominance ratio (vs 2nd place dimension)", 1.5, 5.0, 3.0, 0.1,
    help="The top dimension's dispersion must be at least this many times the runner-up's "
         "to be called a culprit. Below this, the move is treated as platform-wide instead — "
         "everything moved together, so no single segment is to blame.",
)
min_dispersion = st.sidebar.slider(
    "Minimum dispersion to call a culprit", 0.0, 0.1, 0.02, 0.005,
    help="A floor below which even the top dimension is considered noise, not a real culprit — "
         "guards against calling a culprit when nothing actually moved much.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart options")
show_band = st.sidebar.checkbox(
    "Show detection band on trend", value=True,
    help="Shades the ±z·σ corridor the scan uses — anything outside it is what triggers a flag.",
)
show_volume = st.sidebar.checkbox(
    "Overlay request volume", value=False,
    help="Adds a secondary axis showing raw request volume, so you can see at a glance whether "
         "a metric move was a traffic change or a monetisation change.",
)

st.sidebar.markdown("---")

scan_clicked = st.sidebar.button(
    "🔍 Scan for Anomalies", type="primary",
    help="Runs the full trigger scan + culprit ranking against the selected table.",
)

if st.sidebar.button(
    "♻️ Clear cached queries",
    help="Forces every drill-down and LLM report to re-run instead of serving a cached result. "
         "Also the only way to get a fresh Langfuse trace for something already on screen.",
):
    st.cache_data.clear()
    st.toast("Drill-down cache cleared.")

if st.sidebar.button(
    "🆕 Start new session",
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
    if st.button("Refresh model list", width="stretch"):
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
    recheck = st.button("Re-check connection", width="stretch")
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

st.title("🚨 Ad Metrics — Anomaly & Culprit Scanner")
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
                      "dispersion_ratio_threshold": dispersion_ratio_threshold,
                      "min_dispersion": min_dispersion},
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
                                dispersion_ratio_threshold=dispersion_ratio_threshold,
                                min_dispersion=min_dispersion,
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
    # detection band: the +/- z_threshold * sigma corridor the scan actually uses
    if show_band and f"{metric}_mean" in trigger_df.columns:
        mean = float(trigger_df[f"{metric}_mean"].iloc[0])
        std = float(trigger_df[f"{metric}_std"].iloc[0])
        lo, hi = mean - z_threshold * std, mean + z_threshold * std
        fig.add_trace(
            go.Scatter(x=trigger_df["t"], y=[lo] * len(trigger_df), mode="lines",
                       line=dict(width=0), hoverinfo="skip", showlegend=False),
            row=i, col=1, secondary_y=False,
        )
        curve_metric.append(None)
        fig.add_trace(
            go.Scatter(x=trigger_df["t"], y=[hi] * len(trigger_df), mode="lines",
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
            customdata=trigger_df[f"{metric}_z"],
            hovertemplate="%{x}<br>%{y:.4f}<br>z = %{customdata:.2f}<extra></extra>",
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
    "window. The pale corridor is the ±z·σ detection band — anything outside it flags. "
    "**Click any point to inspect that bucket.**"
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
            st.markdown("**Dispersion ranking across all dimensions**")
            st.caption(
                "Bigger bar = that dimension's values disagreed more with each other → more likely "
                "the cause. A dimension where every value moved together is ruled out. "
                "**Click a bar to open that dimension.**"
            )
            bar_fig = px.bar(
                ranking_df.sort_values("dispersion", ascending=True),
                x="dispersion", y="dimension_name", orientation="h",
                color="status",
                color_discrete_map={"Culprit": "#e63946", "Ruled out": "#2a9d8f"},
                text="dispersion",
                custom_data=["min_ratio", "max_ratio", "spread"],
            )
            bar_fig.update_traces(
                texttemplate="%{text:.4f}", textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Dispersion: %{x:.4f}<br>"
                    "Ratio range: %{customdata[0]:.2f}x – %{customdata[1]:.2f}x<br>"
                    "Spread: %{customdata[2]:.4f}<extra></extra>"
                ),
            )
            bar_fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Dispersion (higher = more suspicious)", yaxis_title="",
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
        picked_dim = bar_points[0].get("y") if bar_points else default_dim

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