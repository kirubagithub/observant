import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

from clickhouse_queries import (
    get_client,
    step1_trigger_scan,
    find_incident_windows,
    build_verdict,
    METRIC_LABELS,
)
from llm_stub import generate_llm_diagnosis

load_dotenv()

st.set_page_config(page_title="InMobi Anomaly Scanner", layout="wide", page_icon="🚨")

METRIC_COLORS = {"revenue": "#e63946", "fill_rate": "#f4a261", "ecpm": "#2a9d8f"}

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
st.sidebar.caption(
    f"Connecting to `{os.environ.get('CLICKHOUSE_HOST', 'unset')}` / "
    f"`{os.environ.get('CLICKHOUSE_DATABASE', 'unset')}`"
)

scan_clicked = st.sidebar.button("🔍 Scan for Anomalies", type="primary", use_container_width=True)

st.title("🚨 Ad Metrics — Anomaly & Culprit Scanner")
st.caption(
    "Trigger scan on `dimension_name='__total__'` → dispersion-ranked culprit search across "
    "every dimension → named culprit value. All math runs in ClickHouse SQL; this page only "
    "renders the results (and a placeholder for your own LLM call)."
)

# ---------------------------------------------------------------------------
# Run the pipeline on click
# ---------------------------------------------------------------------------
if scan_clicked:
    with st.spinner("Connecting to ClickHouse and scanning..."):
        try:
            client = get_client()
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

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
n_total = len(verdicts)
n_by_metric = {m: sum(1 for v in verdicts if v["metric"] == m) for m in ("revenue", "fill_rate", "ecpm")}
n_with_culprit = sum(1 for v in verdicts if v["has_culprit"])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total anomalies", n_total)
k2.metric("💰 Revenue", n_by_metric["revenue"])
k3.metric("📶 Fill rate", n_by_metric["fill_rate"])
k4.metric("💵 eCPM", n_by_metric["ecpm"])
k5.metric("🎯 With culprit found", f"{n_with_culprit} / {n_total}" if n_total else "0 / 0")

if n_total > 0:
    donut_col, _ = st.columns([1, 2])
    with donut_col:
        donut_df = pd.DataFrame(
            [{"Metric": METRIC_LABELS[m], "Count": c} for m, c in n_by_metric.items() if c > 0]
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
            st.plotly_chart(donut_fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Trend charts — revenue / fill_rate / ecpm with anomaly windows shaded
# ---------------------------------------------------------------------------
st.subheader("📈 Anomaly Trend — Revenue, Fill Rate & eCPM")

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=("Revenue", "Fill Rate", "eCPM"),
    vertical_spacing=0.08,
)

for i, (metric, col) in enumerate([("revenue", "revenue"), ("fill_rate", "fill_rate"), ("ecpm", "ecpm")], start=1):
    is_anom = trigger_df[f"{metric}_z"].abs() > z_threshold
    fig.add_trace(
        go.Scatter(
            x=trigger_df["date"], y=trigger_df[col], mode="lines+markers",
            name=METRIC_LABELS[metric],
            marker=dict(
                color=["#e63946" if a else "#457b9d" for a in is_anom],
                size=[10 if a else 5 for a in is_anom],
            ),
            line=dict(color="#457b9d"),
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        ),
        row=i, col=1,
    )

for v in verdicts:
    row = {"revenue": 1, "fill_rate": 2, "ecpm": 3}[v["metric"]]
    fig.add_vrect(
        x0=v["window"][0], x1=v["window"][-1],
        fillcolor=METRIC_COLORS[v["metric"]], opacity=0.12, line_width=0,
        row=row, col=1,
    )

fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
st.caption("Red markers = flagged points (|z| above threshold). Shaded band = the grouped incident window for that metric.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader(f"🧾 Anomalies found: {n_total}")

if n_total == 0:
    st.success("No anomalies above the current z-score threshold. Try lowering the threshold in the sidebar.")
else:
    summary_rows = []
    for v in verdicts:
        window_label = f"{v['window'][0]} → {v['window'][-1]}" if len(v["window"]) > 1 else v["window"][0]
        summary_rows.append({
            "Metric": v["metric_label"],
            "Window": window_label,
            "Days": len(v["window"]),
            "Status": "🔴 Culprit found" if v["has_culprit"] else "🟡 Platform-wide (no single culprit)",
            "Culprit": f"{v.get('culprit_dimension', '—')} = {v.get('culprit_value', '—')}" if v["has_culprit"] else "—",
            "Ratio vs baseline": f"{v.get('ratio', 1.0):.2f}x" if v["has_culprit"] else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-incident detail
# ---------------------------------------------------------------------------
st.subheader("🔬 Incident Detail — Culprit Ranking & Diagnosis")


def direction_word(ratio: float) -> str:
    return "dropped" if ratio < 1 else "spiked"


for idx, v in enumerate(verdicts):
    window_label = f"{v['window'][0]} → {v['window'][-1]}" if len(v["window"]) > 1 else v["window"][0]
    header_emoji = "🔴" if v["has_culprit"] else "🟡"

    with st.expander(f"{header_emoji} {v['metric_label']} anomaly — {window_label}", expanded=(idx == 0)):
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
                "the cause. A dimension where every value moved together is ruled out."
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
            )
            st.plotly_chart(bar_fig, use_container_width=True)

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
                st.plotly_chart(gauge_fig, use_container_width=True)
                st.caption(
                    f"Baseline: **{v['baseline_value']}** → During incident: **{v['window_value']}** "
                    f"({v['ratio']:.2f}x, {direction_word(v['ratio'])} {abs(1 - v['ratio']) * 100:.1f}%)"
                )
            else:
                st.markdown("**No single culprit dimension** 🟡")
                st.caption(v.get("note", ""))

        st.markdown("**Ruled out dimensions**")
        ruled_out = v.get("ruled_out", [])
        st.write(" ".join(f"`{d}` ✅" for d in ruled_out) if ruled_out else "—")

        st.markdown("**Verdict payload** (the only thing that should go to an LLM)")
        st.json(v)

        st.markdown("**🧠 Diagnosis**")
        st.caption(
            "This is the output of `llm_stub.generate_llm_diagnosis()`. Replace that function's body "
            "with your own LLM call — whatever string it returns is shown here as page content."
        )
        diagnosis_text = generate_llm_diagnosis(v)
        st.info(diagnosis_text)

st.markdown("---")
st.caption(
    "Source: ClickHouse aggregated table (`dimension_name='__total__'` for trend, per-dimension rows "
    "for culprit ranking) · All detection math runs in SQL · This app only formats results."
)
