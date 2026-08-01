"""
All ClickHouse SQL for the anomaly -> culprit pipeline.

Design principle: ClickHouse does 100% of the math (z-scores, ratios,
dispersion ranking). This module only sends SQL and returns small
DataFrames / dicts. No pandas-side statistics, no row-level LLM payloads.

GRANULARITY: every function below takes a `grain` (see granularity.py)
instead of a hardcoded table name and 'date' column. A grain bundles the
table, the time column, and whether that column is a DATE or a DateTime -
daily and hourly tables are driven by the exact same query text, just with
different SQL fragments substituted in from the grain.
"""

import os
from itertools import groupby

import clickhouse_connect
import numpy as np
import pandas as pd

from granularity import (
    Grain, seasonal_key_expr, seasonal_keys_for_window, seasonal_key_sql_list,
    describe_seasonal_keys, time_literal, time_list_sql, grain_ordinal,
)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

SETTINGS = dict(
    max_execution_time=30,
    max_rows_to_read=1_000_000_000,
    max_bytes_to_read=100_000_000_000,
    timeout_before_checking_execution_speed=0,
)

METRIC_EXPR = {
    "revenue": "revenue",
    "fill_rate": "fills / requests",
    "ecpm": "revenue / impressions * 1000",
}

METRIC_LABELS = {
    "revenue": "Revenue",
    "fill_rate": "Fill Rate",
    "ecpm": "eCPM",
}

# ---------------------------------------------------------------------------
# Numerator / denominator form of each metric — DRILL-DOWNS ONLY.
#
# METRIC_EXPR above (average-of-daily-ratios) still drives detection and the
# verdict, and is deliberately left alone. The drill-downs below use the
# ratio-of-sums form instead, because contribution attribution is only exact
# when numerator and denominator are carried separately:
#
#     R = N / D,  and  contribution_i = (N_iw - R_baseline * D_iw) / D_window
#     sums exactly to (R_window - R_baseline).
#
# CONSEQUENCE: a ratio shown in a drill-down can differ by a few percent from
# the ratio in the verdict above it, because one is volume-weighted and the
# other is not. That gap closes once the detection layer moves to
# ratio-of-sums as well.
# ---------------------------------------------------------------------------

METRIC_NUM = {
    "revenue": "revenue",
    "fill_rate": "fills",
    "ecpm": "revenue * 1000",
}

METRIC_DEN = {
    "revenue": "1",          # sumIf(1, cond) = bucket count -> average per-bucket revenue
    "fill_rate": "requests",
    "ecpm": "impressions",
}

# NOTE: for a "revenue" incident the ranking below uses the `revenue` column
# directly. If you want to disambiguate a pure volume drop (requests fell,
# monetization normal) from a monetization drop (requests normal, eCPM/fill
# fell), extend METRIC_EXPR / build_verdict to also rank dispersion on
# `requests` and compare, as discussed in the anomaly_culprit_pipeline.md doc.


def _lit(value) -> str:
    """Escape a value as a single-quoted SQL string literal.

    Stopgap only. Dimension values come back from ClickHouse and go straight
    into the next query, so they are escaped rather than trusted. Proper
    parameter binding is the real fix and is tracked separately.
    """
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def get_client():
    """Build a ClickHouse client from environment variables."""
    return clickhouse_connect.get_client(
        host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
        port=int(os.environ.get("CLICKHOUSE_PORT", "8443")),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=os.environ.get("CLICKHOUSE_DATABASE", "inmobi"),
        secure=os.environ.get("CLICKHOUSE_SECURE", "true").lower() == "true",
    )


# ---------------------------------------------------------------------------
# Step 1 — trigger scan (cheap: only dimension_name='__total__' rows)
# ---------------------------------------------------------------------------

def step1_trigger_scan(client, grain: Grain) -> pd.DataFrame:
    """The time column is always aliased to `t` in the result, regardless of
    grain, so every downstream consumer (find_incident_windows, the trend
    chart) is grain-agnostic and never has to know whether it is looking at
    a DATE or a DateTime column.
    """
    tc = grain.time_col
    q = f"""
        WITH bucketed AS (
            SELECT {tc} AS t,
                   revenue,
                   requests,
                   fills,
                   impressions,
                   fills / requests             AS fill_rate,
                   revenue / impressions * 1000  AS ecpm
            FROM {grain.table}
            WHERE dimension_name = '__total__'
        )
        SELECT t,
               revenue,
               requests,
               fills,
               impressions,
               fill_rate,
               ecpm,
               (revenue   - avg(revenue)   OVER ()) / stddevPop(revenue)   OVER () AS revenue_z,
               (fill_rate - avg(fill_rate) OVER ()) / stddevPop(fill_rate) OVER () AS fill_rate_z,
               (ecpm      - avg(ecpm)      OVER ()) / stddevPop(ecpm)      OVER () AS ecpm_z,
               avg(revenue)         OVER () AS revenue_mean,
               stddevPop(revenue)   OVER () AS revenue_std,
               avg(fill_rate)       OVER () AS fill_rate_mean,
               stddevPop(fill_rate) OVER () AS fill_rate_std,
               avg(ecpm)            OVER () AS ecpm_mean,
               stddevPop(ecpm)      OVER () AS ecpm_std
        FROM bucketed
        ORDER BY t
    """
    df = client.query_df(q, settings=SETTINGS)
    if df.empty or "t" not in df.columns:
        raise ValueError(
            f"No rows returned from {grain.table} for dimension_name='__total__'. "
            f"Check the table name, the time column ('{tc}'), and that "
            "__total__ rows exist. If this is the hourly table, confirm "
            "CLICKHOUSE_HOURLY_TIME_COL matches the real column name — "
            "'hour' is a guess, not a verified default."
        )
    df["t"] = pd.to_datetime(df["t"])
    if not grain.is_datetime:
        df["t"] = df["t"].dt.date
    return df


def find_incident_windows(df: pd.DataFrame, grain: Grain, z_threshold: float = 2.0) -> dict:
    """Group consecutive flagged buckets per metric into incident windows.

    Returns: {metric: [[t, t, ...], [t, ...], ...]}
    """
    incidents = {}
    for metric in ("revenue", "fill_rate", "ecpm"):
        flagged = sorted(df.loc[df[f"{metric}_z"].abs() > z_threshold, "t"].tolist())
        windows = []
        for _, grp in groupby(enumerate(flagged), lambda x: grain_ordinal(grain, x[1]) - x[0]):
            window = [t for _, t in grp]
            windows.append(window)
        if windows:
            incidents[metric] = windows
    return incidents


# ---------------------------------------------------------------------------
# Step 2 — culprit ranking across ALL dimensions (dispersion-based)
# ---------------------------------------------------------------------------

def step2_culprit_ranking(client, metric: str, window_values, grain: Grain) -> pd.DataFrame:
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    metric_expr = METRIC_EXPR[metric]
    q = f"""
        WITH ratios AS (
            SELECT dimension_name,
                   dimension_value,
                   avgIf({metric_expr}, {tc} IN ({times}))
                 / avgIf({metric_expr}, {tc} NOT IN ({times})) AS ratio
            FROM {grain.table}
            WHERE dimension_name NOT IN ('app_id', '__total__')
            GROUP BY dimension_name, dimension_value
        )
        SELECT dimension_name,
               min(ratio)              AS min_ratio,
               max(ratio)              AS max_ratio,
               max(ratio) - min(ratio) AS spread,
               stddevPop(ratio)        AS dispersion
        FROM ratios
        GROUP BY dimension_name
        ORDER BY dispersion DESC
    """
    return client.query_df(q, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Step 3 — name the exact culprit value within the winning dimension
# ---------------------------------------------------------------------------

def step3_culprit_value(client, metric: str, window_values, dimension_name: str,
                         grain: Grain, ascending: bool = True) -> pd.DataFrame:
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    metric_expr = METRIC_EXPR[metric]
    order = "ASC" if ascending else "DESC"
    q = f"""
        SELECT dimension_value,
               avgIf({metric_expr}, {tc} IN ({times}))     AS window_value,
               avgIf({metric_expr}, {tc} NOT IN ({times})) AS baseline_value,
               avgIf({metric_expr}, {tc} IN ({times}))
             / avgIf({metric_expr}, {tc} NOT IN ({times})) AS ratio
        FROM {grain.table}
        WHERE dimension_name = {_lit(dimension_name)}
        GROUP BY dimension_value
        ORDER BY ratio {order}
    """
    return client.query_df(q, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Orchestration — build one small verdict JSON per incident
# ---------------------------------------------------------------------------

def build_verdict(client, metric: str, window_values, grain: Grain,
                   dispersion_ratio_threshold: float = 3.0, min_dispersion: float = 0.02) -> dict | None:
    ranking_df = step2_culprit_ranking(client, metric, window_values, grain)
    if ranking_df.empty:
        return None

    top = ranking_df.iloc[0]
    second_dispersion = ranking_df.iloc[1]["dispersion"] if len(ranking_df) > 1 else 0.0

    has_culprit = bool(
        top["dispersion"] >= min_dispersion
        and top["dispersion"] >= dispersion_ratio_threshold * max(second_dispersion, 1e-9)
    )

    verdict = {
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "grain": grain.key,
        "window": [time_literal(grain, v).strip("'") for v in window_values],
        "has_culprit": has_culprit,
        "ranking": ranking_df.round(4).to_dict(orient="records"),
    }

    if has_culprit:
        values_df = step3_culprit_value(client, metric, window_values, top["dimension_name"], grain, ascending=True)
        min_row = values_df.iloc[0]
        max_row = values_df.iloc[-1]
        # pick whichever end deviates further from ratio=1 (works for drops and spikes)
        culprit_row = min_row if abs(min_row["ratio"] - 1) >= abs(max_row["ratio"] - 1) else max_row

        verdict.update({
            "culprit_dimension": top["dimension_name"],
            "culprit_value": culprit_row["dimension_value"],
            "window_value": round(float(culprit_row["window_value"]), 4),
            "baseline_value": round(float(culprit_row["baseline_value"]), 4),
            "ratio": round(float(culprit_row["ratio"]), 4),
            "ruled_out": ranking_df["dimension_name"].tolist()[1:],
        })
    else:
        verdict["note"] = "uniform movement across all dimensions — platform-wide, no single segment responsible"
        verdict["ruled_out"] = ranking_df["dimension_name"].tolist()

    return verdict


# ===========================================================================
# DRILL-DOWN QUERIES
#
# None of the functions below feed detection or the verdict. They exist purely
# to answer "why?" once an incident is already on screen. All math stays in
# ClickHouse; these return small DataFrames ready to render.
# ===========================================================================


def drill_dimension_values(client, metric: str, window_values, dimension_name: str,
                           grain: Grain) -> pd.DataFrame:
    """Every value of one dimension: level, ratio, volume and contribution.

    `contribution` is that value's share of the metric's absolute change,
    expressed in metric units. Contributions across all values of a dimension
    sum exactly to (window_value - baseline_value) for the dimension overall,
    which is what makes the waterfall chart add up.
    """
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    num, den = METRIC_NUM[metric], METRIC_DEN[metric]
    q = f"""
        WITH agg AS (
            SELECT dimension_value,
                   sumIf({num}, {tc} IN ({times}))     AS n_w,
                   sumIf({den}, {tc} IN ({times}))     AS d_w,
                   sumIf({num}, {tc} NOT IN ({times})) AS n_b,
                   sumIf({den}, {tc} NOT IN ({times})) AS d_b
            FROM {grain.table}
            WHERE dimension_name = {_lit(dimension_name)}
            GROUP BY dimension_value
        ),
        tot AS (
            SELECT sum(n_w) AS N_W, sum(d_w) AS D_W,
                   sum(n_b) AS N_B, sum(d_b) AS D_B
            FROM agg
        )
        SELECT dimension_value,
               n_w / nullIf(d_w, 0)                                   AS window_value,
               n_b / nullIf(d_b, 0)                                   AS baseline_value,
               (n_w / nullIf(d_w, 0)) / nullIf(n_b / nullIf(d_b, 0), 0) AS ratio,
               d_b                                                    AS baseline_volume,
               d_w                                                    AS window_volume,
               d_b / nullIf(D_B, 0)                                   AS volume_share,
               (n_w - (N_B / nullIf(D_B, 0)) * d_w) / nullIf(D_W, 0)  AS contribution,
               (n_w - (N_B / nullIf(D_B, 0)) * d_w) / nullIf(D_W, 0)
                 / nullIf((N_W / nullIf(D_W, 0)) - (N_B / nullIf(D_B, 0)), 0)
                                                                      AS contribution_share
        FROM agg
        CROSS JOIN tot
        ORDER BY abs(contribution) DESC
    """
    return client.query_df(q, settings=SETTINGS)


def drill_day_top_movers(client, metric: str, bucket_value, grain: Grain,
                          limit: int = 15) -> pd.DataFrame:
    """Top movers on a single bucket (a day, or an hour), across every
    dimension at once.

    Ranked by contribution rather than by raw ratio, so a segment with three
    impressions and a 40x ratio does not outrank a segment that actually moved
    the number. The same underlying shift can appear more than once (an OS
    problem also shows up under device_model) — this is a movers list, not an
    attribution.
    """
    tc = grain.time_col
    t = time_literal(grain, bucket_value)
    num, den = METRIC_NUM[metric], METRIC_DEN[metric]
    q = f"""
        WITH agg AS (
            SELECT dimension_name,
                   dimension_value,
                   sumIf({num}, {tc} =  {t}) AS n_w,
                   sumIf({den}, {tc} =  {t}) AS d_w,
                   sumIf({num}, {tc} != {t}) AS n_b,
                   sumIf({den}, {tc} != {t}) AS d_b
            FROM {grain.table}
            WHERE dimension_name != '__total__'
            GROUP BY dimension_name, dimension_value
        ),
        tot AS (
            SELECT dimension_name,
                   sum(n_w) AS N_W, sum(d_w) AS D_W,
                   sum(n_b) AS N_B, sum(d_b) AS D_B
            FROM agg
            GROUP BY dimension_name
        )
        SELECT a.dimension_name                                           AS dimension_name,
               a.dimension_value                                          AS dimension_value,
               a.n_w / nullIf(a.d_w, 0)                                   AS day_value,
               a.n_b / nullIf(a.d_b, 0)                                   AS baseline_value,
               (a.n_w / nullIf(a.d_w, 0))
                 / nullIf(a.n_b / nullIf(a.d_b, 0), 0)                    AS ratio,
               a.d_w                                                      AS day_volume,
               (a.n_w - (t.N_B / nullIf(t.D_B, 0)) * a.d_w)
                 / nullIf(t.D_W, 0)                                       AS contribution
        FROM agg AS a
        INNER JOIN tot AS t ON a.dimension_name = t.dimension_name
        ORDER BY abs(contribution) DESC
        LIMIT {int(limit)}
    """
    return client.query_df(q, settings=SETTINGS)


def drill_segment_timeseries(client, metric: str, dimension_name: str, dimension_value: str,
                              grain: Grain) -> pd.DataFrame:
    """Per-bucket series for one segment alongside the platform total.

    Answers the questions the gauge cannot: when did it start, was it a step or
    a ramp, and did it come back.
    """
    tc = grain.time_col
    num, den = METRIC_NUM[metric], METRIC_DEN[metric]
    dim, val = _lit(dimension_name), _lit(dimension_value)
    q = f"""
        SELECT {tc} AS t,
               sumIf({num}, dimension_name = {dim} AND dimension_value = {val})
             / nullIf(sumIf({den}, dimension_name = {dim} AND dimension_value = {val}), 0)
                   AS segment_value,
               sumIf({num}, dimension_name = '__total__')
             / nullIf(sumIf({den}, dimension_name = '__total__'), 0)
                   AS total_value
        FROM {grain.table}
        WHERE (dimension_name = {dim} AND dimension_value = {val})
           OR dimension_name = '__total__'
        GROUP BY t
        ORDER BY t
    """
    df = client.query_df(q, settings=SETTINGS)
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"])
        if not grain.is_datetime:
            df["t"] = df["t"].dt.date
    return df


def drill_revenue_decomposition(client, window_values, grain: Grain) -> pd.DataFrame:
    """Split a revenue move into its four multiplicative drivers.

    Exact identity from the dataset glossary:

        revenue = requests x (fills/requests) x (impressions/fills) x (eCPM/1000)

    In log space the four factors add up, so the bars sum exactly to the total
    log change in average per-bucket revenue.

    Returns one row per driver plus a `residual` attribute check. Built as a
    single-row SELECT reshaped in pandas rather than a four-branch UNION ALL:
    ClickHouse resolves a trailing ORDER BY against the last branch of a union,
    so an alias defined only in the first branch is not visible and the query
    fails with UNKNOWN_IDENTIFIER. One row, four columns, no union, no ordering
    problem — and the CTE is scanned once instead of four times.
    """
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    q = f"""
        WITH t AS (
            SELECT sumIf(requests,    {tc} IN ({times}))     AS req_w,
                   sumIf(fills,       {tc} IN ({times}))     AS fil_w,
                   sumIf(impressions, {tc} IN ({times}))     AS imp_w,
                   sumIf(revenue,     {tc} IN ({times}))     AS rev_w,
                   countIf({tc} IN ({times}))                AS nd_w,
                   sumIf(requests,    {tc} NOT IN ({times})) AS req_b,
                   sumIf(fills,       {tc} NOT IN ({times})) AS fil_b,
                   sumIf(impressions, {tc} NOT IN ({times})) AS imp_b,
                   sumIf(revenue,     {tc} NOT IN ({times})) AS rev_b,
                   countIf({tc} NOT IN ({times}))            AS nd_b
            FROM {grain.table}
            WHERE dimension_name = '__total__'
        )
        SELECT
            log((req_w / nullIf(nd_w, 0))  / nullIf(req_b / nullIf(nd_b, 0), 0))  AS d_requests,
            log((fil_w / nullIf(req_w, 0)) / nullIf(fil_b / nullIf(req_b, 0), 0)) AS d_fill_rate,
            log((imp_w / nullIf(fil_w, 0)) / nullIf(imp_b / nullIf(fil_b, 0), 0)) AS d_render_rate,
            log((rev_w / nullIf(imp_w, 0)) / nullIf(rev_b / nullIf(imp_b, 0), 0)) AS d_ecpm,
            log((rev_w / nullIf(nd_w, 0))  / nullIf(rev_b / nullIf(nd_b, 0), 0))  AS d_total
        FROM t
    """
    raw = client.query_df(q, settings=SETTINGS)
    if raw.empty:
        return pd.DataFrame(columns=["factor", "log_delta", "pct_change"])

    row = raw.iloc[0]
    factors = [
        ("Requests (volume)", "d_requests"),
        ("Fill rate", "d_fill_rate"),
        ("Render rate (imps/fill)", "d_render_rate"),
        ("eCPM", "d_ecpm"),
    ]
    df = pd.DataFrame(
        [{"factor": label, "log_delta": float(row[col]) if pd.notna(row[col]) else np.nan}
         for label, col in factors]
    )
    df["pct_change"] = np.expm1(df["log_delta"]) * 100

    # The four drivers must sum to the total. If they do not, a denominator was
    # zero somewhere and the chart would silently mislead.
    total = float(row["d_total"]) if pd.notna(row["d_total"]) else np.nan
    df.attrs["log_total"] = total
    df.attrs["residual"] = (
        total - float(np.nansum(df["log_delta"])) if pd.notna(total) else np.nan
    )
    return df


def drill_onset_heatmap(client, metric: str, window_values, dimension_name: str,
                         grain: Grain, top_n: int = 15) -> pd.DataFrame:
    """Ratio-vs-own-baseline per dimension value per bucket.

    Restricted to the top N values by baseline volume, otherwise the long tail
    of tiny segments dominates the colour scale.
    """
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    num, den = METRIC_NUM[metric], METRIC_DEN[metric]
    dim = _lit(dimension_name)
    q = f"""
        WITH base AS (
            SELECT dimension_value,
                   sumIf({num}, {tc} NOT IN ({times}))
                 / nullIf(sumIf({den}, {tc} NOT IN ({times})), 0) AS b,
                   sumIf({den}, {tc} NOT IN ({times}))            AS vol
            FROM {grain.table}
            WHERE dimension_name = {dim}
            GROUP BY dimension_value
            ORDER BY vol DESC
            LIMIT {int(top_n)}
        ),
        bucketed AS (
            SELECT dimension_value,
                   {tc} AS t,
                   sum({num}) / nullIf(sum({den}), 0) AS v
            FROM {grain.table}
            WHERE dimension_name = {dim}
            GROUP BY dimension_value, t
        )
        SELECT d.dimension_value AS dimension_value,
               d.t               AS t,
               d.v / nullIf(b.b, 0) AS ratio
        FROM bucketed AS d
        INNER JOIN base AS b ON d.dimension_value = b.dimension_value
        ORDER BY d.dimension_value, d.t
    """
    df = client.query_df(q, settings=SETTINGS)
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"])
        if not grain.is_datetime:
            df["t"] = df["t"].dt.date
    return df


# ===========================================================================
# EVIDENCE FOR THE REPORT LLM
#
# The headline numbers the report quotes must be platform-level (__total__),
# not the culprit segment's. build_verdict only ever computed the segment's,
# so these are new. Ratio metrics use sum/sum per the dataset glossary.
# ===========================================================================

def incident_headline(client, metric: str, window_values, grain: Grain) -> dict:
    """Platform-level move for the driving metric, plus a same-season baseline.

    Returns current_value, baseline_value, delta_pct (flat baseline) and
    seasonal_baseline_value / seasonal_delta_pct (same seasonal-key buckets
    only — same weekday for daily, same weekday+hour for hourly).

    The glossary for this dataset is explicit that the data carries weekly
    seasonality with lower weekends, that a flat global average makes every
    weekend look anomalous, and that at least one planted movement is pure
    seasonality that should be ruled out rather than alarmed on. Hourly data
    layers an intraday cycle on top of that same weekly one, which is why the
    seasonal key is two-dimensional (weekday, hour) at that grain instead of
    weekday alone.
    """
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    keys = seasonal_keys_for_window(grain, window_values)
    key_expr = seasonal_key_expr(grain)
    key_list = seasonal_key_sql_list(grain, keys)
    num, den = METRIC_NUM[metric], METRIC_DEN[metric]
    q = f"""
        SELECT sumIf({num}, {tc} IN ({times}))
             / nullIf(sumIf({den}, {tc} IN ({times})), 0)                  AS current_value,
               sumIf({num}, {tc} NOT IN ({times}))
             / nullIf(sumIf({den}, {tc} NOT IN ({times})), 0)              AS baseline_value,
               sumIf({num}, {tc} NOT IN ({times}) AND {key_expr} IN {key_list})
             / nullIf(sumIf({den}, {tc} NOT IN ({times})
                                  AND {key_expr} IN {key_list}), 0) AS seasonal_baseline_value
        FROM {grain.table}
        WHERE dimension_name = '__total__'
    """
    df = client.query_df(q, settings=SETTINGS)
    if df.empty:
        return {}

    row = df.iloc[0]
    cur = row["current_value"]
    flat = row["baseline_value"]
    seas = row["seasonal_baseline_value"]
    if pd.isna(cur) or pd.isna(flat) or not flat:
        return {}

    out = {
        "current_value": float(cur),
        "baseline_value": float(flat),
        "delta_pct": (float(cur) / float(flat) - 1.0) * 100.0,
        "grain": grain.key,
        "seasonal_keys": keys,
    }
    if not pd.isna(seas) and seas:
        out["seasonal_baseline_value"] = float(seas)
        out["seasonal_delta_pct"] = (float(cur) / float(seas) - 1.0) * 100.0
    return out


def seasonality_note(headline: dict, grain: Grain, shrink_threshold: float = 0.5) -> str | None:
    """One factual sentence comparing the flat and same-season baselines.

    Returns None when there is nothing to say, so the field is omitted from the
    payload rather than padded. Every number in the returned string comes from
    `headline`; none is estimated.
    """
    if not headline or "seasonal_delta_pct" not in headline:
        return None

    flat = headline["delta_pct"]
    seasonal = headline["seasonal_delta_pct"]
    when = describe_seasonal_keys(grain, headline.get("seasonal_keys", []))
    season_word = "same-weekday" if not grain.is_datetime else "same-hour-of-week"

    if abs(flat) < 1e-9:
        return None

    # how much of the flat move survives a like-for-like comparison
    if abs(seasonal) <= shrink_threshold * abs(flat):
        return (
            f"Window falls on {when}; against a {season_word} baseline the move is only "
            f"{seasonal:+.1f}% versus {flat:+.1f}% flat, so most of it is seasonality "
            f"({grain.min_history_hint})."
        )
    return (
        f"Window falls on {when}; against a {season_word} baseline the move is still "
        f"{seasonal:+.1f}% versus {flat:+.1f}% flat, so seasonality does not explain it."
    )