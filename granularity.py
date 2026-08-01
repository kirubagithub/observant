"""
Daily vs hourly table selection.

Two tables, same conceptual schema, different time grain. Everything in this
module is config plus small SQL-fragment builders; the actual detection and
drill-down queries in clickhouse_queries.py take a `time_col` and `is_hourly`
argument rather than hardcoding 'date'.

WHAT I DO NOT KNOW AND YOU MUST VERIFY:
The hourly table's real column name and type. I have never seen its schema.
CLICKHOUSE_HOURLY_TIME_COL defaults to a guess ("hour"). If your table uses a
different name (`event_hour`, `ts`, `timestamp`, ...) or a different type
(DateTime vs DateTime64 vs a truncated date), set CLICKHOUSE_HOURLY_TIME_COL in
.env before switching the app to Hourly, or every query in this file will fail
with an "Unknown identifier" error identical in shape to the decomposition bug
from earlier - same root cause, wrong column name, different file.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Grain:
    key: str                 # "daily" | "hourly" - internal, never shown in the UI
    label: str                # "Daily" | "Hourly" - what the sidebar shows
    table: str
    time_col: str
    is_datetime: bool         # False = DATE column, True = DateTime/DateTime64
    bucket_seconds: int        # size of one row's time bucket, for window math
    min_history_hint: str      # shown in the UI when there may not be enough
                               # history for the seasonal baseline to be reliable


def _grain(key: str, label: str, table_env: str, default_table: str,
          time_col_env: str, default_time_col: str, is_datetime: bool,
          bucket_seconds: int, min_history_hint: str) -> Grain:
    return Grain(
        key=key,
        label=label,
        table=os.environ.get(table_env, default_table),
        time_col=os.environ.get(time_col_env, default_time_col),
        is_datetime=is_datetime,
        bucket_seconds=bucket_seconds,
        min_history_hint=min_history_hint,
    )


def load_grains() -> dict:
    """Read table/column config fresh from the environment.

    A function, not a module-level constant, so sanitize_env() and any .env
    reload happening earlier in app startup are reflected.
    """
    return {
        "daily": _grain(
            "daily", "Daily",
            "CLICKHOUSE_TABLE_DAILY", "ad_events_daily_agg",
            "CLICKHOUSE_DAILY_TIME_COL", "date",
            is_datetime=False,
            bucket_seconds=86400,
            min_history_hint="needs several weeks of history for a reliable same-weekday baseline",
        ),
        "hourly": _grain(
            "hourly", "Hourly",
            "CLICKHOUSE_TABLE_HOURLY", "ad_events_hourly_agg",
            "CLICKHOUSE_HOURLY_TIME_COL", "hour",  # GUESS - confirm against your schema
            is_datetime=True,
            bucket_seconds=3600,
            min_history_hint="needs several weeks of history for a reliable same-hour-of-week baseline "
                             "(24x more rows than daily for the same calendar span)",
        ),
    }


# ---------------------------------------------------------------------------
# Seasonal-key SQL fragments
#
# Daily seasonality is "same weekday." Hourly seasonality is stronger and
# two-dimensional - intraday (night trough vs afternoon peak) AND weekly (the
# glossary's weekend effect still applies at hourly grain, layered on top of
# the daily cycle). A flat hourly baseline would flag every night as an
# incident the same way a flat daily baseline flagged every weekend.
#
# The seasonal key is what "same time as this window" means for a grain:
#   daily:  weekday only              -> toDayOfWeek(time_col)
#   hourly: (weekday, hour) pair      -> (toDayOfWeek(time_col), toHour(time_col))
# ---------------------------------------------------------------------------

def seasonal_key_expr(grain: Grain) -> str:
    """SQL expression computing this row's seasonal key."""
    if grain.is_datetime:
        return f"(toDayOfWeek({grain.time_col}), toHour({grain.time_col}))"
    return f"toDayOfWeek({grain.time_col})"


def seasonal_keys_for_window(grain: Grain, window_values) -> list:
    """Python-side: the distinct seasonal keys actually present in a window.

    window_values are date objects (daily) or datetime objects (hourly).
    Returned as a list of ints (daily) or (weekday, hour) tuples (hourly), for
    both building the SQL IN-list and for describing the window in plain
    English (e.g. "Mon/Tue/Wed" or "Mon 14:00, Tue 15:00").
    """
    if grain.is_datetime:
        return sorted({(v.isoweekday(), v.hour) for v in window_values})
    return sorted({v.isoweekday() for v in window_values})


def seasonal_key_sql_list(grain: Grain, keys) -> str:
    """Render seasonal keys as a SQL literal for use in an IN (...) clause.

    Daily: a plain int list -> IN (1,2,3)
    Hourly: an array-of-tuples -> IN [(1,14),(2,9)] - ClickHouse's tuple-array
    IN syntax, not standard SQL; verify against your ClickHouse version if this
    errors, as it is less commonly used than scalar IN lists.
    """
    if grain.is_datetime:
        return "[" + ",".join(f"({w},{h})" for w, h in keys) + "]"
    return "(" + ",".join(str(k) for k in keys) + ")"


_WEEKDAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def describe_seasonal_keys(grain: Grain, keys) -> str:
    """Plain-English rendering for the seasonality_note sentence."""
    if grain.is_datetime:
        return ", ".join(f"{_WEEKDAY_NAMES[w]} {h:02d}:00" for w, h in keys)
    return "/".join(_WEEKDAY_NAMES[k] for k in keys)


def time_literal(grain: Grain, value) -> str:
    """A window value as a SQL literal matching the column type."""
    if grain.is_datetime:
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    return f"'{value.isoformat()}'"


def time_list_sql(grain: Grain, values) -> str:
    return ",".join(time_literal(grain, v) for v in values)


def grain_ordinal(grain: Grain, value) -> int:
    """Integer that increases by exactly 1 between consecutive buckets.

    Used to group consecutive flagged buckets into one incident window,
    regardless of whether a bucket is a calendar day or an hour. Daily uses
    the proleptic Gregorian ordinal directly; hourly divides Unix time by the
    bucket size. Absolute values differ between the two - only the "+1 per
    step" property is required, and only within one grain at a time.
    """
    if grain.is_datetime:
        return int(value.timestamp() // grain.bucket_seconds)
    return value.toordinal()