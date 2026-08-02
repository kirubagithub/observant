# InMobi Anomaly & Culprit Scanner (Streamlit)

Dockerized Streamlit app that:

1. Runs the **trigger scan** SQL against `dimension_name='__total__'` in
   `ad_events_daily_agg` to find days where revenue / fill_rate / eCPM
   moved beyond a z-score threshold.
2. For each flagged window, runs the **dispersion-ranking** SQL across
   every other dimension (`country`, `os_version`, `category`, `ad_format`,
   `device_model`, `region`, `publisher_tier`, `campaign_type`, `vertical`)
   to find which one, if any, is the culprit.
3. Names the exact **culprit value** (e.g. `os_version = Android 15`).
4. Renders everything — trend charts, dispersion bar charts, gauges, a
   summary table, and the small JSON verdict — directly on the page.
5. Calls a **stub LLM function** (`llm_stub.generate_llm_diagnosis`) per
   incident and prints whatever string it returns as page content. Replace
   that function's body with your own LLM API call — you don't need to
   touch anything else in the app.

All statistics (z-scores, ratios, dispersion) are computed in ClickHouse
SQL, not in Python/pandas. The Python layer only orchestrates queries and
renders results.

## Run with Docker Compose (recommended)

```bash
cp example.env .env
# edit .env with your ClickHouse Cloud host / user / password / database

docker compose up --build
```

Open http://localhost:8501, click **Scan for Anomalies** in the sidebar.

## Run with plain Docker

```bash
cp example.env .env
# edit .env

docker build -t inmobi-anomaly-scanner .
docker run --rm -p 8501:8501 --env-file .env inmobi-anomaly-scanner
```

## Run locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp example.env .env   # edit it
streamlit run app.py
```

## Use the provided schema to ingest data

This repository includes `schemas.sql`, which creates the raw event table,
lookup tables, dictionary sources, hourly and daily aggregate tables, and
materialized views that populate those aggregates automatically.

The basic ingestion flow is:

1. Create or choose the target ClickHouse database (for example `inmobi_cat`).
2. Run the schema file against that database:

```bash
clickhouse-client \
  --host "$CLICKHOUSE_HOST" --port "$CLICKHOUSE_PORT" \
  --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" \
  --secure=1 \
  --database "$CLICKHOUSE_DATABASE" \
  < schemas.sql
```

3. Load raw event rows into `inmobi_cat.ad_events`.
4. Populate the lookup tables: `inmobi_cat.apps`, `inmobi_cat.advertisers`, and
   `inmobi_cat.geo_device`.
5. The materialized views will aggregate those rows into
   `inmobi_cat.ad_events_daily_agg` and `inmobi_cat.ad_events_hourly_agg`.

If you already have your own aggregated table, set `CLICKHOUSE_TABLE` or
`CLICKHOUSE_TABLE_DAILY` / `CLICKHOUSE_TABLE_HOURLY` to point the app at it.

## Configuration

All connection settings come from environment variables (loaded from `.env`
via `python-dotenv`, or injected by Docker/Compose):

| Variable                | Meaning                                        | Default              |
|-------------------------|-----------------------------------------------|-----------------------|
| `CLICKHOUSE_HOST`       | ClickHouse Cloud host                         | `localhost`           |
| `CLICKHOUSE_PORT`       | HTTPS port                                    | `8443`                |
| `CLICKHOUSE_USER`       | Username                                      | `default`             |
| `CLICKHOUSE_PASSWORD`   | Password                                      | *(empty)*             |
| `CLICKHOUSE_DATABASE`   | Database containing the agg table             | `inmobi`              |
| `CLICKHOUSE_SECURE`     | Use TLS (`true`/`false`)                      | `true`                |
| `CLICKHOUSE_TABLE`      | Daily aggregate table name (editable in UI too)| `ad_events_daily_agg` |
| `CLICKHOUSE_TABLE_DAILY` | Daily aggregate table name used by the daily grain | `ad_events_daily_agg` |
| `CLICKHOUSE_DAILY_TIME_COL` | Date column name for the daily table | `date` |
| `CLICKHOUSE_TABLE_HOURLY` | Hourly aggregate table name used by the hourly grain | `ad_events_hourly_agg` |
| `CLICKHOUSE_HOURLY_TIME_COL` | DateTime column name for the hourly table | `hour` |

The sidebar can switch between Daily and Hourly grains. If you use the hourly table, set `CLICKHOUSE_TABLE_HOURLY` and `CLICKHOUSE_HOURLY_TIME_COL` in `.env` to match your schema.

Notes:

- `CLICKHOUSE_TABLE` is the default table name used by the daily grain unless `CLICKHOUSE_TABLE_DAILY` is set.
- `CLICKHOUSE_TABLE_DAILY` and `CLICKHOUSE_DAILY_TIME_COL` explicitly configure the daily grain.
- `CLICKHOUSE_TABLE_HOURLY` and `CLICKHOUSE_HOURLY_TIME_COL` explicitly configure the hourly grain.

Sidebar sliders (no restart needed):

- **Anomaly z-score threshold** — how extreme a day must be vs. the rest of
  the period to count as an anomaly (default 2.0).
- **Culprit dominance ratio** — how much higher the top dimension's
  dispersion must be vs. the runner-up to call it a real culprit, not noise
  (default 3.0x).
- **Minimum dispersion to call a culprit** — floor below which even the top
  dimension is considered "everything moved together" (default 0.02).

## Where to plug in your own LLM call

Open `llm_stub.py`. Everything you need is already built:

- `build_prompt(verdict)` turns the verdict JSON into the exact prompt text.
- `generate_llm_diagnosis(verdict)` is called once per incident from
  `app.py` and its **return value is rendered as page content** in an
  `st.info(...)` box. Replace the placeholder body with a real API call
  (OpenAI, Anthropic, local model, etc.) — no other file needs to change.

```python
def generate_llm_diagnosis(verdict: dict) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(verdict)}],
        max_tokens=120,
    )
    return resp.choices[0].message.content
```

Because the verdict JSON is small (~10 fields, no raw rows), this call
should be cheap — a few hundred tokens per incident, not per row.

## File layout

```
anomaly-streamlit-app/
├── app.py                  # Streamlit UI: KPIs, trend charts, per-incident detail
├── clickhouse_queries.py   # All SQL: trigger scan, dispersion ranking, culprit naming
├── llm_stub.py             # <-- wire your own LLM call here
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example → provided as `example.env` (copy to `.env`)
└── README.md
```
