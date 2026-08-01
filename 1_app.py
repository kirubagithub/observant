import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

import drilldowns as dd
from clickhouse_queries import (
    get_client,
    step1_trigger_scan,
    find_incident_windows,
    build_verdict,
    METRIC_LABELS,
)

load_dotenv()

st.set_page_config(page_title="InMobi Anomaly Scanner", layout="wide", page_icon="🚨")

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

table = st.sidebar.text_input(
    "Daily agg table",
    value=os.environ.get("CLICKHOUSE_TABLE", "ad_events_daily_agg"),
)
z_threshold = st.sidebar.slider("Anomaly z-score threshold", 1.5, 4.0, 2.0, 0.1)
dispersion_ratio_threshold = st.sidebar.slider(
    "Culprit dominance ratio (vs 2nd place dimension)", 1.5, 5.0, 3.0, 0.1
)
min_dispersion = st.sidebar.slider(
    "Minimum dispersion to call a culprit", 0.0, 0.1, 0.02, 0.005
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Chart options")
show_band = st.sidebar.checkbox("Show detection band on trend", value=True)
show_volume = st.sidebar.checkbox("Overlay request volume", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Connecting to `{os.environ.get('CLICKHOUSE_HOST', 'unset')}` / "
    f"`{os.environ.get('CLICKHOUSE_DATABASE', 'unset')}`"
)

scan_clicked = st.sidebar.button("🔍 Scan for Anomalies", type="primary", use_container_width=True)

if st.sidebar.button("♻️ Clear cached queries", use_container_width=True):
    st.cache_data.clear()
    st.toast("Drill-down cache cleared.")

st.title("🚨 Ad Metrics — Anomaly & Culprit Scanner")
st.caption(
    "Trigger scan on `dimension_name='__total__'` → dispersion-ranked culprit search across "
    "every dimension → named culprit value. All math runs in ClickHouse SQL; this page only "
    "renders the results (and a placeholder for your own LLM call). "
    "**Charts are clickable** — click a trend point or a dispersion bar to drill in."
)

# ---------------------------------------------------------------------------
# Run the pipeline on click
# ---------------------------------------------------------------------------
if scan_clicked:
    with st.spinner("Connecting to ClickHouse and scanning..."):
        try:
            client = get_cached_client()
            trigger_df = step1_trigger_scan(client, table=table)
            incidents = find_incident_windows(trigger_df, z_threshold=z_threshold)

            verdicts = []
            for metric, windows in incidents.items():
                for window_dates in windows:
                    v = build_verdict(
                        client, metric, window_dates, table=table,
                        dispersion_ratio_threshold=dispersion_ratio_threshold,
                        min_dispersion=min_dispersion,
                    )
                    if v:
                        verdicts.append(v)

            st.session_state["trigger_df"] = trigger_df
            st.session_state["verdicts"] = verdicts
            st.session_state["z_threshold"] = z_threshold
            st.session_state["table"] = table
            st.session_state["scanned"] = True
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.stop()

if not st.session_state.get("scanned"):
    st.info("👈 Click **Scan for Anomalies** in the sidebar to run the pipeline.")
    st.stop()

trigger_df = st.session_state["trigger_df"]
verdicts = st.session_state["verdicts"]
z_threshold = st.session_state["z_threshold"]
scan_table = st.session_state.get("table", table)
client = get_cached_client()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
n_total = len(verdicts)
n_by_metric = {m: sum(1 for v in verdicts if v["metric"] == m) for m in METRIC_ORDER}
n_with_culprit = sum(1 for v in verdicts if v["has_culprit"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total anomalies", n_total)
k2.metric("💰 Revenue", n_by_metric["revenue"])
k3.metric("📶 Fill rate", n_by_metric["fill_rate"])
k4.metric("💵 eCPM", n_by_metric["ecpm"])
k5.metric("🎯 With culprit found", f"{n_with_culprit} / {n_total}" if n_total else "0 / 0")

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
            st.plotly_chart(donut_fig, use_container_width=True, key="donut")
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
st.subheader("📈 Anomaly Trend — Revenue, Fill Rate & eCPM")

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
            go.Scatter(x=trigger_df["date"], y=[lo] * len(trigger_df), mode="lines",
                       line=dict(width=0), hoverinfo="skip", showlegend=False),
            row=i, col=1, secondary_y=False,
        )
        curve_metric.append(None)
        fig.add_trace(
            go.Scatter(x=trigger_df["date"], y=[hi] * len(trigger_df), mode="lines",
                       line=dict(width=0), fill="tonexty", fillcolor="rgba(69,123,157,0.10)",
                       hoverinfo="skip", showlegend=False),
            row=i, col=1, secondary_y=False,
        )
        curve_metric.append(None)

    is_anom = trigger_df[f"{metric}_z"].abs() > z_threshold
    fig.add_trace(
        go.Scatter(
            x=trigger_df["date"], y=trigger_df[metric], mode="lines+markers",
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
            go.Scatter(x=trigger_df["date"], y=trigger_df["requests"], mode="lines",
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
    fig, use_container_width=True, key="trend_chart", on_select="rerun"
)
st.caption(
    "Red markers = flagged points (|z| above threshold). Shaded band = the grouped incident "
    "window. The pale corridor is the ±z·σ detection band — anything outside it flags. "
    "**Click any point to inspect that day.**"
)

points = dd._selected_points(trend_event)
if points:
    pt = points[0]
    picked_metric = curve_metric[pt.get("curve_number", 0)] if pt.get("curve_number", 0) < len(curve_metric) else None
    if picked_metric is None:
        st.info("That trace isn't drillable — click a coloured metric point instead.")
    else:
        picked_day = pd.to_datetime(pt["x"]).date()
        with st.container(border=True):
            dd.render_day_panel(client, picked_metric, picked_day, scan_table, key_prefix="trend")
else:
    st.caption("👆 No day selected. Click a point on any of the three charts above.")

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
        summary_rows.append({
            "Metric": v["metric_label"],
            "Window": window_label,
            "Days": len(v["window"]),
            "Status": "🔴 Culprit found" if v["has_culprit"] else "🟡 Platform-wide (no single culprit)",
            "Culprit": f"{v.get('culprit_dimension', '—')} = {v.get('culprit_value', '—')}" if v["has_culprit"] else "—",
            "Ratio vs baseline": f"{v.get('ratio', 1.0):.2f}x" if v["has_culprit"] else "—",
        })
    summary_event = st.dataframe(
        pd.DataFrame(summary_rows), use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="summary_table",
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

    with st.expander(f"{header_emoji} {v['metric_label']} anomaly — {window_label}", expanded=(idx == focus_idx)):
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
            )
            bar_fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            bar_fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Dispersion (higher = more suspicious)", yaxis_title="",
                clickmode="event+select",
            )
            bar_event = st.plotly_chart(
                bar_fig, use_container_width=True, key=f"bar_{idx}", on_select="rerun"
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
                st.plotly_chart(gauge_fig, use_container_width=True, key=f"gauge_{idx}")
                st.caption(
                    f"Baseline: **{v['baseline_value']}** → During incident: **{v['window_value']}** "
                    f"({v['ratio']:.2f}x, {direction_word(v['ratio'])} {abs(1 - v['ratio']) * 100:.1f}%)"
                )
            else:
                st.markdown("**No single culprit dimension** 🟡")
                st.caption(v.get("note", ""))

        # --- dimension drill-down, driven by the bar click ------------------
        window_dates = [pd.to_datetime(d).date() for d in v["window"]]
        bar_points = dd._selected_points(bar_event)
        default_dim = v.get("culprit_dimension") or (ranking_df["dimension_name"].iloc[0] if not ranking_df.empty else None)
        picked_dim = bar_points[0].get("y") if bar_points else default_dim

        if picked_dim:
            with st.container(border=True):
                dd.render_dimension_panel(client, v, str(picked_dim), scan_table, key_prefix=f"inc{idx}")

        # --- revenue-only: volume vs monetisation decomposition -------------
        if v["metric"] == "revenue":
            with st.container(border=True):
                dd.render_revenue_decomposition(client, window_dates, scan_table, key_prefix=f"inc{idx}")

        st.markdown("**Ruled out dimensions**")
        ruled_out = v.get("ruled_out", [])
        st.write(" ".join(f"`{d}` ✅" for d in ruled_out) if ruled_out else "—")

        st.markdown("**Verdict payload** (the only thing that should go to an LLM)")
        st.json(v)

        st.markdown("**🧠 Diagnosis**")
        st.caption(
            "This is the output of `llm_stub.generate_llm_diagnosis()`. Replace that function's body "
            "with your own LLM call — whatever string it returns is shown here as page content. "
            "Cached on the verdict, so drill-down clicks do not re-trigger it."
        )
        diagnosis_text = dd.cached_diagnosis(dd.verdict_key(v), v)
        st.info(diagnosis_text)

st.markdown("---")
st.caption(
    "Source: ClickHouse aggregated table (`dimension_name='__total__'` for trend, per-dimension rows "
    "for culprit ranking) · All detection math runs in SQL · This app only formats results."
)