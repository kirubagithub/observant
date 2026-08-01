"""
All ClickHouse SQL for the anomaly -> culprit pipeline.

Design principle: ClickHouse does 100% of the math (z-scores, ratios,
dispersion ranking). This module only sends SQL and returns small
DataFrames / dicts. No pandas-side statistics, no row-level LLM payloads.
"""

import os
from itertools import groupby

import clickhouse_connect
import pandas as pd

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

# NOTE: for a "revenue" incident the ranking below uses the `revenue` column
# directly. If you want to disambiguate a pure volume drop (requests fell,
# monetization normal) from a monetization drop (requests normal, eCPM/fill
# fell), extend METRIC_EXPR / build_verdict to also rank dispersion on
# `requests` and compare, as discussed in the anomaly_culprit_pipeline.md doc.


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

def step1_trigger_scan(client, table: str = "ad_events_daily_agg") -> pd.DataFrame:
    q = f"""
        WITH daily AS (
            SELECT date,
                   revenue,
                   fills / requests             AS fill_rate,
                   revenue / impressions * 1000  AS ecpm
            FROM {table}
            WHERE dimension_name = '__total__'
        )
        SELECT date,
               revenue,
               fill_rate,
               ecpm,
               (revenue   - avg(revenue)   OVER ()) / stddevPop(revenue)   OVER () AS revenue_z,
               (fill_rate - avg(fill_rate) OVER ()) / stddevPop(fill_rate) OVER () AS fill_rate_z,
               (ecpm      - avg(ecpm)      OVER ()) / stddevPop(ecpm)      OVER () AS ecpm_z
        FROM daily
        ORDER BY date
    """
    df = client.query_df(q, settings=SETTINGS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def find_incident_windows(df: pd.DataFrame, z_threshold: float = 2.0) -> dict:
    """Group consecutive flagged dates per metric into incident windows.

    Returns: {metric: [[date, date, ...], [date, ...], ...]}
    """
    incidents = {}
    for metric in ("revenue", "fill_rate", "ecpm"):
        flagged = sorted(df.loc[df[f"{metric}_z"].abs() > z_threshold, "date"].tolist())
        windows = []
        for _, grp in groupby(enumerate(flagged), lambda x: x[1].toordinal() - x[0]):
            window = [d for _, d in grp]
            windows.append(window)
        if windows:
            incidents[metric] = windows
    return incidents


# ---------------------------------------------------------------------------
# Step 2 — culprit ranking across ALL dimensions (dispersion-based)
# ---------------------------------------------------------------------------

def step2_culprit_ranking(client, metric: str, window_dates, table: str = "ad_events_daily_agg") -> pd.DataFrame:
    date_list = ",".join(f"'{d.isoformat()}'" for d in window_dates)
    metric_expr = METRIC_EXPR[metric]
    q = f"""
        WITH ratios AS (
            SELECT dimension_name,
                   dimension_value,
                   avgIf({metric_expr}, date IN ({date_list}))
                 / avgIf({metric_expr}, date NOT IN ({date_list})) AS ratio
            FROM {table}
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

def step3_culprit_value(client, metric: str, window_dates, dimension_name: str,
                         table: str = "ad_events_daily_agg", ascending: bool = True) -> pd.DataFrame:
    date_list = ",".join(f"'{d.isoformat()}'" for d in window_dates)
    metric_expr = METRIC_EXPR[metric]
    order = "ASC" if ascending else "DESC"
    q = f"""
        SELECT dimension_value,
               avgIf({metric_expr}, date IN ({date_list}))     AS window_value,
               avgIf({metric_expr}, date NOT IN ({date_list})) AS baseline_value,
               avgIf({metric_expr}, date IN ({date_list}))
             / avgIf({metric_expr}, date NOT IN ({date_list})) AS ratio
        FROM {table}
        WHERE dimension_name = '{dimension_name}'
        GROUP BY dimension_value
        ORDER BY ratio {order}
    """
    return client.query_df(q, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Orchestration — build one small verdict JSON per incident
# ---------------------------------------------------------------------------

def build_verdict(client, metric: str, window_dates, table: str = "ad_events_daily_agg",
                   dispersion_ratio_threshold: float = 3.0, min_dispersion: float = 0.02) -> dict | None:
    ranking_df = step2_culprit_ranking(client, metric, window_dates, table)
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
        "window": [d.isoformat() for d in window_dates],
        "has_culprit": has_culprit,
        "ranking": ranking_df.round(4).to_dict(orient="records"),
    }

    if has_culprit:
        values_df = step3_culprit_value(client, metric, window_dates, top["dimension_name"], table, ascending=True)
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
