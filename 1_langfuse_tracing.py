"""
Langfuse tracing for the ClickHouse pipeline (scans and drill-downs) —
separate from llm_stub.py, which owns the Groq/report tracing.

Design:
- Reuses llm_stub's already-verified Langfuse connection (_init_langfuse) so
  there is exactly one place that decides whether tracing is active, not two
  drifting copies of the same check.
- Every public function here is a no-op that returns something safely
  chainable if tracing is off or the SDK throws — a tracing failure must never
  break a ClickHouse query. Same principle as the rest of this app: the
  primary function (showing data) is never sacrificed for the secondary one
  (observing it).
- One Langfuse trace per user action (one scan, one drill-down render), with
  the underlying SQL calls as child spans nested inside it via OTel context.
  Every trace in one browser tab shares a session_id, so Langfuse's Sessions
  view groups a whole page-load's worth of exploration together.
- Deliberately NOT wired into clickhouse_queries.py. That module's own
  docstring says it "only sends SQL and returns small DataFrames" — tracing
  concerns live here, at the call site, so that module stays swappable/
  testable without any Langfuse dependency.

WHAT GETS TRACED VS WHAT DOESN'T:
Drill-down queries are wrapped where they are actually called — inside the
@st.cache_data-decorated functions in drilldowns.py. Those bodies only run on
a genuine cache miss, so tracing there means "a trace represents a real
ClickHouse call," not "a trace represents something the user saw on screen."
Same gap that already exists for the LLM report cache, for the same reason:
correctness of what a trace means matters more than trace count matching
page content. Clearing the query cache (sidebar button) forces every visible
drill-down to re-run and therefore re-trace.
"""

import uuid
from contextlib import contextmanager

import streamlit as st


def get_session_id() -> str:
    """One id per browser tab's Streamlit session, stable across reruns."""
    if "langfuse_session_id" not in st.session_state:
        st.session_state["langfuse_session_id"] = f"streamlit-{uuid.uuid4().hex[:16]}"
    return st.session_state["langfuse_session_id"]


def new_session() -> str:
    """Explicitly start a new session, without waiting for a browser refresh.

    A genuinely new browser tab already gets a fresh st.session_state (and
    therefore a fresh session_id) for free — Streamlit gives every new
    WebSocket connection its own session_state from scratch. This function is
    for the case that free reset doesn't cover: staying in the SAME tab and
    wanting a clean slate on demand (e.g. after finishing a review pass) without
    a hard reload or process restart.
    """
    st.session_state["langfuse_session_id"] = f"streamlit-{uuid.uuid4().hex[:16]}"
    return st.session_state["langfuse_session_id"]


def current_ids() -> dict:
    """Trace/span id + dashboard URL for whatever traced()/traced_root() block
    is currently open. Call this INSIDE the `with` block — both ids are only
    valid while their span is active; there is nothing to read once it closes.
    Returns empty values (not an exception) when tracing is off, so callers
    can render "no trace" state without a try/except at every call site.
    """
    client = _tracing_available()
    if client is None:
        return {"trace_id": None, "span_id": None, "trace_url": None}
    try:
        trace_id = client.get_current_trace_id()
        span_id = client.get_current_observation_id()
        url = client.get_trace_url(trace_id=trace_id) if trace_id else None
        return {"trace_id": trace_id, "span_id": span_id, "trace_url": url}
    except Exception:
        return {"trace_id": None, "span_id": None, "trace_url": None}


class _NoOpSpan:
    """Returned when tracing is off or fails to init. Every real span method
    used at call sites must have a matching no-op here so callers never need
    an `if traced:` branch."""

    def update(self, **kwargs):
        pass


def _tracing_available():
    """Reuse llm_stub's single source of truth for "is Langfuse actually up."

    Import is deferred (not at module top) so this module can be imported even
    in contexts where llm_stub's Groq-specific env vars are not set yet.
    """
    try:
        from llm_stub import _init_langfuse, _LANGFUSE_STATE
        if _init_langfuse():
            return _LANGFUSE_STATE["client"]
    except Exception:
        pass
    return None


@contextmanager
def traced_root(name: str, input: dict | None = None, metadata: dict | None = None):
    """One trace per user action (one scan, one drill-down render).

    All nested `traced()` spans opened inside this context automatically
    become children of it, and inherit the session_id via propagate_attributes.
    """
    client = _tracing_available()
    if client is None:
        yield _NoOpSpan()
        return

    try:
        from langfuse import propagate_attributes
        with propagate_attributes(session_id=get_session_id()):
            with client.start_as_current_observation(
                name=name, as_type="span", input=input, metadata=metadata
            ) as span:
                yield span
    except Exception:
        # A tracing-library error must never take down the actual query path.
        yield _NoOpSpan()


@contextmanager
def traced(name: str, input: dict | None = None, metadata: dict | None = None):
    """A child span. Nests under whatever traced_root (or another traced())
    is currently open; if none is open, Langfuse makes it its own root.
    """
    client = _tracing_available()
    if client is None:
        yield _NoOpSpan()
        return

    try:
        with client.start_as_current_observation(
            name=name, as_type="span", input=input, metadata=metadata
        ) as span:
            yield span
    except Exception:
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Output summarisation
#
# Same principle as the LLM evidence trimming: a span's output is for
# understanding what happened, not for reproducing the full result. A drill-
# down DataFrame can be dozens of rows; log its shape and a small sample, not
# the whole thing.
# ---------------------------------------------------------------------------

def summarize_df(df, sample_rows: int = 5) -> dict:
    """A DataFrame -> a small dict safe to attach as span output."""
    try:
        if df is None or df.empty:
            return {"rows": 0}
        return {
            "rows": int(len(df)),
            "columns": list(df.columns)[:12],
            "sample": df.head(sample_rows).to_dict(orient="records"),
        }
    except Exception:
        return {"rows": None}