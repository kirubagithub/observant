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
    Grain, seasonal_key_expr, day_type_key_expr, seasonal_keys_for_window,
    seasonal_key_sql_list, describe_seasonal_keys, time_literal, time_list_sql,
    grain_ordinal,
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

def step1_trigger_scan(client, grain: Grain,
                       min_seasonal_samples: int = 8,
                       min_daytype_samples: int = 8) -> pd.DataFrame:
    """The time column is always aliased to `t` in the result, regardless of
    grain, so every downstream consumer (find_incident_windows, the trend
    chart) is grain-agnostic and never has to know whether it is looking at
    a DATE or a DateTime column.

    SEASONAL BASELINE, NOT FLAT: mean/std/percentiles are all partitioned by
    the grain's seasonal key (same weekday for daily; same weekday+hour for
    hourly) rather than computed once over the whole period. This is the
    fix for the exact failure mode the dataset glossary names directly —
    "a flat global average makes every weekend look like an anomaly" — by
    making the trigger's own baseline seasonal, not by adding an explanatory
    note after a false alarm has already fired.

    THREE-TIER FALLBACK — this matters, and a two-tier version of this
    (seasonal, else flat) is NOT enough. Measured directly: on 5 weeks of
    data with a real ~25% weekend dip, a flat fallback's std comes out 2.4x
    too wide, PURELY from mixing weekday and weekend values that don't
    belong in the same distribution — reintroducing the exact contamination
    problem seasonal partitioning exists to fix, at the fallback layer. So
    there are three tiers, each used only when there's enough history to
    trust it, falling back progressively rather than jumping straight to
    the contaminated flat estimate:

        1. Full seasonal (weekday, or weekday+hour for hourly) — finest,
           needs `min_seasonal_samples` per bucket.
        2. Weekday-vs-weekend only — coarser, needs `min_daytype_samples`
           per bucket, but still isolates the single biggest seasonal
           effect instead of averaging over it.
        3. Flat (whole period) — only if even tier 2 doesn't have enough
           samples. Still seasonality-contaminated; this is a last resort,
           not a design goal.

    `baseline_tier` reports which one applied per row ('seasonal' /
    'day_type' / 'flat'), so the UI can disclose it rather than presenting
    every row's band with the same implied confidence.

    PERCENTILES ARE APPROXIMATE: `quantile()` (not `quantileExact()`) trades
    a small, bounded error for speed on large tables — reasonable for a
    monitoring band, not for a number you'd cite precisely. The same
    three-tier fallback applies to them.
    """
    tc = grain.time_col
    # The outer SELECT below operates on `bucketed`, which aliases the time
    # column to `t` — so every key expression must reference `t`, not
    # grain's original column name (that identifier no longer exists here).
    key_expr = seasonal_key_expr(grain, column="t")
    daytype_expr = day_type_key_expr(grain, column="t")

    def stat(fn: str, col: str) -> str:
        """`fn` computed at all three tiers, cascaded by sample size."""
        seasonal = f"{fn}({col}) OVER (PARTITION BY {key_expr})"
        daytype = f"{fn}({col}) OVER (PARTITION BY {daytype_expr})"
        flat = f"{fn}({col}) OVER ()"
        return (
            f"if(bucket_n >= {min_seasonal_samples}, {seasonal}, "
            f"if(daytype_n >= {min_daytype_samples}, {daytype}, {flat}))"
        )

    q = f"""
        WITH bucketed AS (
            SELECT {tc} AS t,
                   revenue,
                   requests,
                   fills,
                   impressions,
                   fills / requests             AS fill_rate,
                   revenue / impressions * 1000  AS ecpm,
                   count() OVER (PARTITION BY {key_expr})     AS bucket_n,
                   count() OVER (PARTITION BY {daytype_expr}) AS daytype_n
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
               bucket_n,
               daytype_n,
               if(bucket_n >= {min_seasonal_samples}, 'seasonal',
                  if(daytype_n >= {min_daytype_samples}, 'day_type', 'flat')) AS baseline_tier,

               (revenue   - {stat("avg", "revenue")})   / nullIf({stat("stddevPop", "revenue")}, 0)   AS revenue_z,
               (fill_rate - {stat("avg", "fill_rate")}) / nullIf({stat("stddevPop", "fill_rate")}, 0) AS fill_rate_z,
               (ecpm      - {stat("avg", "ecpm")})      / nullIf({stat("stddevPop", "ecpm")}, 0)      AS ecpm_z,

               {stat("avg", "revenue")}         AS revenue_mean,
               {stat("stddevPop", "revenue")}   AS revenue_std,
               {stat("avg", "fill_rate")}       AS fill_rate_mean,
               {stat("stddevPop", "fill_rate")} AS fill_rate_std,
               {stat("avg", "ecpm")}            AS ecpm_mean,
               {stat("stddevPop", "ecpm")}      AS ecpm_std,

               {stat("quantile(0.10)", "revenue")} AS revenue_p10,
               {stat("quantile(0.90)", "revenue")} AS revenue_p90,
               {stat("quantile(0.95)", "revenue")} AS revenue_p95,
               {stat("quantile(0.99)", "revenue")} AS revenue_p99,

               {stat("quantile(0.10)", "fill_rate")} AS fill_rate_p10,
               {stat("quantile(0.90)", "fill_rate")} AS fill_rate_p90,
               {stat("quantile(0.95)", "fill_rate")} AS fill_rate_p95,
               {stat("quantile(0.99)", "fill_rate")} AS fill_rate_p99,

               {stat("quantile(0.10)", "ecpm")} AS ecpm_p10,
               {stat("quantile(0.90)", "ecpm")} AS ecpm_p90,
               {stat("quantile(0.95)", "ecpm")} AS ecpm_p95,
               {stat("quantile(0.99)", "ecpm")} AS ecpm_p99
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

def step2_culprit_ranking(client, metric: str, window_values, grain: Grain,
                          min_volume_share: float = 0.01) -> pd.DataFrame:
    """Rank every dimension by its single best explanation for the observed
    change — not by how spread-out the whole dimension is.

    WHY THE OLD APPROACH (dispersion = stddevPop of per-value ratios) WAS
    WRONG: it answers "how much does this dimension disagree with itself,"
    not "is there one value that broke while the rest are normal." Verified
    directly against a real incident: os_version had one catastrophic value
    (a fill-rate collapse, z=-10.1 relative to the other 7 OS versions, all
    of which were normal) but LOST the ranking to publisher_tier — which had
    all 3 of its values move by modest, DIFFERENT amounts (a mix shift, not
    a single culprit). Dispersion rewarded the mix shift over the actual
    single-value break, because stddev across only 3 values is much easier
    to inflate than stddev diluted across 8 values where 7 are quiet.

    THE FIX: contribution_share — what fraction of the TOTAL observed change
    in the metric one value accounts for, using the exact attribution
    identity already used in the drill-down waterfall (contribution sums
    exactly to the total change, so shares are directly comparable across
    dimensions of any cardinality). Ranking dimensions by their single
    highest |contribution_share| value fixes the low-cardinality bias:
    Android 12 alone explaining ~90%+ of a fill-rate drop outranks three
    publisher tiers each explaining a modest, separate slice.

    VOLUME FLOOR (`min_volume_share`, default 1% of the dimension's own
    baseline volume): without this, a near-empty segment can produce a wild
    ratio and thus a wild contribution from noise alone. This is a
    heuristic default, not a rigorously derived one — tune it against your
    actual segment volume distribution.

    Returns one row per dimension: the descriptive dispersion stats (kept
    for continuity/display) PLUS `top_value`, `top_contribution_share`,
    `top_window_value`, `top_baseline_value`, `top_ratio` — everything
    needed to name a culprit directly, without a separate step3 query.
    """
    tc = grain.time_col
    times = time_list_sql(grain, window_values)
    metric_expr = METRIC_EXPR[metric]
    num, den = METRIC_NUM[metric], METRIC_DEN[metric]

    # CONFIRMED BUG, fixed here: vertical/campaign_type are only populated on
    # FILLED events (per the dataset glossary — advertiser_id, and therefore
    # vertical/campaign_type, is empty on unfilled requests). For fill_rate
    # specifically, the denominator is ALL requests including unfilled ones,
    # so grouping fill_rate by vertical/campaign_type mixes a numerator that
    # only exists for filled rows against a denominator that doesn't respect
    # that split. Verified directly against real data: both dimensions
    # returned dispersion=0 and an empty/degenerate row for a real fill_rate
    # incident — not "no signal," a broken computation. Excluded here rather
    # than silently producing a meaningless zero that could be misread as
    # "definitely ruled out."
    excluded_dims = ["'app_id'", "'__total__'"]
    if metric == "fill_rate":
        excluded_dims += ["'vertical'", "'campaign_type'"]
    excluded_sql = ", ".join(excluded_dims)

    q = f"""
        WITH per_value AS (
            SELECT dimension_name,
                   dimension_value,
                   avgIf({metric_expr}, {tc} IN ({times}))
                 / avgIf({metric_expr}, {tc} NOT IN ({times})) AS ratio,
                   sumIf({num}, {tc} IN ({times}))     AS n_w,
                   sumIf({den}, {tc} IN ({times}))     AS d_w,
                   sumIf({num}, {tc} NOT IN ({times})) AS n_b,
                   sumIf({den}, {tc} NOT IN ({times})) AS d_b
            FROM {grain.table}
            WHERE dimension_name NOT IN ({excluded_sql})
            GROUP BY dimension_name, dimension_value
        ),
        totals AS (
            SELECT dimension_name,
                   sum(n_w) AS N_W, sum(d_w) AS D_W,
                   sum(n_b) AS N_B, sum(d_b) AS D_B
            FROM per_value
            GROUP BY dimension_name
        ),
        scored AS (
            SELECT p.dimension_name                                          AS dimension_name,
                   p.dimension_value                                         AS dimension_value,
                   p.ratio                                                   AS ratio,
                   p.n_w / nullIf(p.d_w, 0)                                  AS window_value,
                   p.n_b / nullIf(p.d_b, 0)                                  AS baseline_value,
                   (p.n_w - (t.N_B / nullIf(t.D_B, 0)) * p.d_w) / nullIf(t.D_W, 0)
                                                                             AS contribution,
                   (p.n_w - (t.N_B / nullIf(t.D_B, 0)) * p.d_w) / nullIf(t.D_W, 0)
                     / nullIf((t.N_W / nullIf(t.D_W, 0)) - (t.N_B / nullIf(t.D_B, 0)), 0)
                                                                             AS contribution_share
            FROM per_value AS p
            INNER JOIN totals AS t ON p.dimension_name = t.dimension_name
            WHERE p.d_b >= {min_volume_share} * t.D_B   -- volume floor, per dimension
        )
        SELECT dimension_name,
               min(ratio)              AS min_ratio,
               max(ratio)              AS max_ratio,
               max(ratio) - min(ratio) AS spread,
               stddevPop(ratio)        AS dispersion,
               max(abs(contribution_share))                           AS top_contribution_share,
               argMax(dimension_value, abs(contribution_share))       AS top_value,
               argMax(window_value, abs(contribution_share))         AS top_window_value,
               argMax(baseline_value, abs(contribution_share))       AS top_baseline_value,
               argMax(ratio, abs(contribution_share))                AS top_ratio
        FROM scored
        GROUP BY dimension_name
        ORDER BY top_contribution_share DESC
    """
    return client.query_df(q, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Step 3 — name the exact culprit value within the winning dimension
# ---------------------------------------------------------------------------

def step3_culprit_value(client, metric: str, window_values, dimension_name: str,
                         grain: Grain, ascending: bool = True) -> pd.DataFrame:
    """Superseded by step2_culprit_ranking's argMax columns — the new ranking
    names the top value directly, so build_verdict no longer calls this.
    Left in place in case you want the full ranked list of every value in
    one dimension independent of a verdict (e.g. for a standalone drill-down
    that isn't tied to build_verdict's decision)."""
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
                   min_contribution_share: float = 0.5, min_volume_share: float = 0.01) -> dict | None:
    """Name a culprit directly from the contribution-share ranking.

    Replaces the old dispersion_ratio_threshold/min_dispersion pair with one
    that means what it says: a culprit is declared when the single best
    (dimension, value) pair across the whole scan explains at least
    `min_contribution_share` of the total observed change. Default 0.5 —
    "this one segment accounts for at least half the move" — is a starting
    point, not a derived constant; tune it against your own data's noise
    floor the same way the old threshold needed tuning.

    Because step2_culprit_ranking now returns the top value directly
    (via argMax), there's no separate step3 query needed here — the old
    two-step "rank dimensions, then look inside the winner" became
    unnecessary once ranking is done on (dimension, value) contribution
    rather than dimension-level dispersion.
    """
    ranking_df = step2_culprit_ranking(client, metric, window_values, grain, min_volume_share)
    if ranking_df.empty:
        return None

    top = ranking_df.iloc[0]
    has_culprit = bool(top["top_contribution_share"] >= min_contribution_share)

    verdict = {
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "grain": grain.key,
        "window": [time_literal(grain, v).strip("'") for v in window_values],
        "has_culprit": has_culprit,
        "ranking": ranking_df.round(4).to_dict(orient="records"),
    }

    if has_culprit:
        verdict.update({
            "culprit_dimension": top["dimension_name"],
            "culprit_value": top["top_value"],
            "window_value": round(float(top["top_window_value"]), 4),
            "baseline_value": round(float(top["top_baseline_value"]), 4),
            "ratio": round(float(top["top_ratio"]), 4),
            "contribution_share": round(float(top["top_contribution_share"]), 4),
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