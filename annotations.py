"""
Incident annotations — durable, unlike everything in st.session_state.

Acknowledging an incident only means something if it survives past the
current browser session. st.session_state is deliberately the wrong place
for this: it is supposed to evaporate on a fresh session (see the session
lifecycle note in app.py), and annotations must not.

Storage: SQLite, one row per unique incident fingerprint. Good enough for a
single-instance deployment; if this app ever runs multiple replicas behind a
load balancer, swap this for the actual ClickHouse database (a tiny table,
`INSERT ... ON DUPLICATE`-style upsert) instead of a local file — a SQLite
file is not shared between containers.

PERSISTENCE CAVEAT: the SQLite file lives at ANNOTATIONS_DB_PATH (default
./annotations.db, i.e. wherever the process's working directory is). In
Docker, that path is inside the container's writable layer, which is wiped
on every `docker compose down` / rebuild unless you mount a volume over it.
Add a volume in docker-compose.yml before relying on this in production.
"""

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager

STATUSES = ("New", "Investigating", "Resolved", "False positive")

# Default is a relative path so a plain `streamlit run app.py` (no Docker)
# works with zero configuration — it lands in whatever directory the process
# was started from. Docker users should override this to a mounted volume
# path in .env (docker-compose.yml already does); local users can leave it.
DB_PATH = os.environ.get("ANNOTATIONS_DB_PATH", "annotations.db")


def fingerprint(grain_key: str, verdict: dict) -> str:
    """A stable id for "this specific detected incident."

    Built from what makes an incident the same incident on re-scan: the
    grain, metric, exact window, and culprit (or the no-culprit case). Two
    different windows, or two different culprits for the same metric, get
    different fingerprints on purpose — annotating one should not silently
    annotate a different incident that happens to share a metric name.
    """
    window = "|".join(verdict.get("window", []))
    culprit = f"{verdict.get('culprit_dimension')}={verdict.get('culprit_value')}" \
        if verdict.get("has_culprit") else "no-culprit"
    raw = f"{grain_key}:{verdict.get('metric')}:{window}:{culprit}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@contextmanager
def _conn():
    # sqlite3.connect raises "unable to open database file" if the parent
    # directory doesn't exist — it does NOT create it for you. This bit
    # whether run locally (a relative path, parent = cwd, always exists) or
    # in Docker (ANNOTATIONS_DB_PATH pointing at a mounted volume that may
    # not have been created yet, e.g. because the container was restarted
    # rather than recreated after the volume was added to docker-compose.yml).
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                fingerprint   TEXT PRIMARY KEY,
                grain         TEXT,
                metric        TEXT,
                window_start  TEXT,
                window_end    TEXT,
                culprit       TEXT,
                status        TEXT NOT NULL DEFAULT 'New',
                note          TEXT NOT NULL DEFAULT '',
                updated_at    REAL,
                updated_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get(fp: str) -> dict | None:
    """Fetch one annotation by fingerprint, or None if never annotated."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT status, note, updated_at, updated_count FROM annotations WHERE fingerprint = ?",
            (fp,),
        ).fetchone()
    if not row:
        return None
    status, note, updated_at, updated_count = row
    return {"status": status, "note": note, "updated_at": updated_at, "updated_count": updated_count}


def set_annotation(fp: str, grain_key: str, verdict: dict, status: str, note: str) -> None:
    """Upsert. updated_count increments so the UI can show 'seen 3 times'
    for a recurring incident rather than just the latest note."""
    window = verdict.get("window", [])
    culprit = f"{verdict.get('culprit_dimension')}={verdict.get('culprit_value')}" \
        if verdict.get("has_culprit") else None
    with _conn() as conn:
        existing = conn.execute(
            "SELECT updated_count FROM annotations WHERE fingerprint = ?", (fp,)
        ).fetchone()
        count = (existing[0] if existing else 0) + 1
        conn.execute("""
            INSERT INTO annotations
                (fingerprint, grain, metric, window_start, window_end, culprit,
                 status, note, updated_at, updated_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                status = excluded.status,
                note = excluded.note,
                updated_at = excluded.updated_at,
                updated_count = excluded.updated_count
        """, (
            fp, grain_key, verdict.get("metric"),
            window[0] if window else None, window[-1] if window else None,
            culprit, status, note, time.time(), count,
        ))


def all_annotations() -> list[dict]:
    """Every annotated incident, most recently updated first. Backs a future
    'incident history' view — not wired into the UI yet, but the data is
    already durable and queryable."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT fingerprint, grain, metric, window_start, window_end, culprit, "
            "status, note, updated_at, updated_count FROM annotations "
            "ORDER BY updated_at DESC"
        ).fetchall()
    cols = ("fingerprint", "grain", "metric", "window_start", "window_end",
            "culprit", "status", "note", "updated_at", "updated_count")
    return [dict(zip(cols, r)) for r in rows]


# ===========================================================================
# CROSS-INCIDENT CORRELATION
#
# In-memory only, computed fresh from one scan's verdicts — not persisted,
# unlike the annotations above. "These two incidents share a culprit" is a
# fact about THIS scan's results, not a durable judgement someone made, so it
# doesn't belong in the same store.
# ===========================================================================

def correlate(verdicts: list[dict]) -> dict:
    """Group incident indices by shared (culprit_dimension, culprit_value).

    Returns {(dimension, value): [indices into `verdicts`]} for every culprit
    that appears in more than one incident. A single os_version breaking both
    fill_rate and revenue in the same scan is the common real case this
    catches — same underlying cause, two metrics, easy to miss if you only
    look at incidents one at a time.
    """
    groups: dict[tuple, list[int]] = {}
    for i, v in enumerate(verdicts):
        if not v.get("has_culprit"):
            continue
        key = (v["culprit_dimension"], v["culprit_value"])
        groups.setdefault(key, []).append(i)
    return {k: idxs for k, idxs in groups.items() if len(idxs) > 1}