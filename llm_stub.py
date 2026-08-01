"""
Report generation — YOU wire the real API call in here.

The rest of the app already did the heavy lifting in ClickHouse. This module
receives a trimmed evidence payload (see llm_payload.build_evidence) and returns
a three-key report dict:

    {"top_level_summary": ..., "root_cause_localization": ..., "checked_and_ruled_out": ...}

Two things are deliberate:

1. The instruction block is a SEPARATE, byte-stable system message and the
   evidence is the user message. Nothing in the system message varies per
   incident. Groq's docs state that cached tokens do not count toward rate
   limits, so a stable prefix is worth more than any further payload trimming —
   the instructions are the larger half of each request.

2. The function never raises. A 429, a timeout or malformed JSON returns a
   fallback report built from the evidence, because the return value is rendered
   straight into the page and an exception here would break the incident loop.
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# Static instruction block — keep byte-stable so it stays cacheable
# ---------------------------------------------------------------------------

REPORT_KEYS = ("top_level_summary", "root_cause_localization", "checked_and_ruled_out")

# Trimmed for token cost: same three rules, same three keys, no pretty-printed
# JSON example (the model does not need indentation to produce compact JSON;
# it needs the key names, which are stated directly). Roughly a third shorter
# than the original by character count. Content requirements are unchanged -
# this is a compression pass, not a behavior change.
SYSTEM_PROMPT = (
    "You write root-cause reports for an ad-tech observability system from a "
    "small JSON verdict computed by deterministic SQL (window, driving_metric, "
    "current_value, baseline_value, delta_pct, and either culprit_dimension/"
    "culprit_value/culprit_delta_pct or a uniform-movement note, plus "
    "ruled_out_dimensions and optionally seasonality_note).\n\n"
    "Return ONLY a JSON object with exactly these three string keys, no other "
    "text, no code fences: top_level_summary, root_cause_localization, "
    "checked_and_ruled_out.\n\n"
    "top_level_summary: one sentence stating what moved (driving_metric), "
    "quoting current_value, baseline_value and delta_pct exactly as given.\n"
    "root_cause_localization: one sentence. If has_culprit is true, name "
    "culprit_dimension, culprit_value and culprit_delta_pct. If false, say the "
    "move was platform-wide using the note field, not one dimension.\n"
    "checked_and_ruled_out: one sentence listing ruled_out_dimensions as "
    "checked and cleared; include seasonality_note if present.\n\n"
    "Rules: use only numbers present in the payload, rounding is fine but "
    "never invent or estimate one; never speculate beyond the evidence; plain "
    "language, no markdown, one paragraph per key."
)


def build_messages(evidence_text: str) -> list[dict]:
    """Static system prompt + variable evidence. Order matters for prefix caching."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": evidence_text},
    ]


# Kept for backward compatibility with the previous single-string interface.
def build_prompt(evidence: dict) -> str:
    from llm_payload import serialise
    return SYSTEM_PROMPT + "\n\nJSON:\n" + serialise(evidence, "json_compact")


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------

def parse_report(raw: str) -> dict:
    """Parse the model's reply into the three-key report. Raises on bad shape."""
    text = raw.strip()
    if text.startswith("```"):                      # strip fences if the model adds them
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    report = json.loads(text)
    missing = [k for k in REPORT_KEYS if k not in report]
    if missing:
        raise ValueError(f"model reply missing keys: {missing}")
    return {k: str(report[k]).strip() for k in REPORT_KEYS}


def fallback_report(evidence: dict, reason: str = "") -> dict:
    """Deterministic report built only from the evidence. No API key needed.

    This is also what the app renders until you wire a real call in, so the UI
    is fully testable offline.
    """
    metric = evidence.get("driving_metric", "The metric")
    window = evidence.get("window", "the incident window")
    prefix = f"[{reason}] " if reason else "[PLACEHOLDER - wire your LLM call in llm_stub.py] "

    if "delta_pct" in evidence:
        summary = (
            f"{prefix}{metric} moved {evidence['delta_pct']:+.1f}% over {window}, from a baseline "
            f"of {evidence['baseline_value']} to {evidence['current_value']}."
        )
    else:
        summary = f"{prefix}{metric} moved anomalously over {window}."

    if evidence.get("has_culprit"):
        localization = (
            f"The move localises to {evidence.get('culprit_dimension')} = "
            f"{evidence.get('culprit_value')}, which changed "
            f"{evidence.get('culprit_delta_pct', 0):+.1f}% against its own baseline."
        )
    else:
        localization = (
            "No single dimension explains the move: "
            + evidence.get("note", "it was uniform across all segments.")
        )

    ruled = ", ".join(evidence.get("ruled_out_dimensions", [])) or "none"
    checked = f"Checked and cleared: {ruled}."
    if evidence.get("seasonality_note"):
        checked += " " + evidence["seasonality_note"]

    return {
        "top_level_summary": summary,
        "root_cause_localization": localization,
        "checked_and_ruled_out": checked,
    }


# ---------------------------------------------------------------------------
# The call itself
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Live call — Groq for inference, Langfuse for tracing
#
# Degradation ladder, most capable first:
#   1. Langfuse-wrapped OpenAI client -> traced Groq call
#   2. Plain OpenAI client            -> untraced Groq call (langfuse missing/misconfigured)
#   3. fallback_report()              -> no network at all
# Nothing here raises. The return value is rendered straight into the page.
# ---------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Values that must never carry stray quote characters. python-dotenv strips
# surrounding quotes, but a value quoted inside docker-compose, or double-quoted
# in .env, arrives with literal " characters attached. Langfuse hands
# LANGFUSE_BASE_URL straight to the OTel exporter, which then tries to resolve
# a URL beginning with a quote and fails with "No connection adapters were
# found for '\"https://...\"'". Stripping them at startup fixes it at source.
_QUOTED_ENV_KEYS = (
    "LANGFUSE_BASE_URL", "LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
    "LANGFUSE_DEBUG",
    "GROQ_API_KEY", "GROQ_MODEL",
    "CLICKHOUSE_HOST", "CLICKHOUSE_DATABASE", "CLICKHOUSE_TABLE",
    "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD",
)


def sanitize_env() -> None:
    """Strip wrapping quotes and whitespace from environment values.

    Call once at startup, before any SDK reads its configuration.
    """
    for key in _QUOTED_ENV_KEYS:
        raw = os.environ.get(key)
        if raw is None:
            continue
        cleaned = raw.strip().strip('"').strip("'").strip()
        if cleaned != raw:
            os.environ[key] = cleaned

# No hardcoded default. Groq disables models per project, so any ID picked in
# advance is a guess. The app asks the account what it will actually serve and
# picks the smallest chat model from that list. GROQ_MODEL, if set, wins.
DEFAULT_MODEL = None

# Not chat models — audio, safety classifiers, embeddings.
_NON_CHAT = ("whisper", "guard", "tts", "embed", "playai", "rerank")

# Ranked preference. Summarising a ten-field JSON object into three sentences is
# an easy task; the smallest enabled model is the right one, and it keeps the
# most headroom under the 6,000 TPM free-tier cap.
_PREFERRED = ("instant", "8b", "9b", "mini", "gemma", "scout", "20b", "70b")

_RESOLVED = {"model": None}

MAX_ATTEMPTS = 3

# Groq disables models at the project level by default; a blocked model returns
# 403 rather than 404, and retrying will never help.
_BLOCKED_MARKERS = ("model_permission_blocked", "blocked at the project level",
                    "permissions_error", "model_not_found", "does not exist")


def chat_models(ids: list[str]) -> list[str]:
    """Drop non-chat models, then sort smallest-looking first."""
    usable = [m for m in ids if not any(x in m.lower() for x in _NON_CHAT)]

    def rank(model_id: str) -> tuple[int, str]:
        low = model_id.lower()
        for i, hint in enumerate(_PREFERRED):
            # Size hints need a digit boundary: a plain substring test makes
            # "120b" match "20b" and rank a 120B model as small.
            pattern = rf"(?<![0-9]){re.escape(hint)}" if hint[0].isdigit() else re.escape(hint)
            if re.search(pattern, low):
                return (i, model_id)
        return (len(_PREFERRED), model_id)

    return sorted(usable, key=rank)


def resolve_model(force: bool = False, exclude: str | None = None) -> str | None:
    """The model to actually call.

    Order: GROQ_MODEL if set -> cached resolution -> ask the account and pick the
    smallest enabled chat model. Returns None if nothing can be resolved.

    The ranking is a name-substring heuristic, not a capability lookup — Groq
    does not publish parameter counts through this endpoint. It biases toward
    small models, which is what this task wants.
    """
    sanitize_env()
    configured = os.environ.get("GROQ_MODEL")
    if configured and configured != exclude and not force:
        return configured

    if _RESOLVED["model"] and _RESOLVED["model"] != exclude and not force:
        return _RESOLVED["model"]

    ok, result = list_models()
    if not ok or not result:
        return None

    candidates = [m for m in chat_models(result) if m != exclude]
    _RESOLVED["model"] = candidates[0] if candidates else None
    return _RESOLVED["model"]


def list_models() -> tuple[bool, object]:
    """Return (True, [model ids]) or (False, human-readable reason).

    Groq's project settings block models by default, so the ID you configured may
    be valid and still refused. This asks the account which models it will
    actually serve, rather than guessing from documentation.
    """
    sanitize_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return False, "GROQ_API_KEY is not set."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        return True, sorted(m.id for m in client.models.list().data)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"


# Confirmed against the installed SDK (langfuse/_client/client.py): base_url
# resolves as `base_url arg > LANGFUSE_BASE_URL env > host arg > LANGFUSE_HOST
# env > cloud default`. Both env vars work; LANGFUSE_BASE_URL wins if both are
# set, and is the current name — LANGFUSE_HOST is documented as deprecated.
#
# The bigger gotcha is not the variable name, it is timing. `get_client()` -
# which `langfuse.openai`'s monkey-patch calls internally on every request -
# builds its Langfuse client ONCE and caches it as a singleton keyed by
# public_key ("get_client.py": "uses a singleton pattern ... to conserve
# resources and maintain state"). In a long-running Streamlit process, if that
# first call ever happened with missing or malformed credentials — including
# before you added them to .env, or before sanitize_env() existed — the
# singleton is stuck wrong until the PROCESS restarts. A page rerun (R key or
# browser refresh) does NOT restart the process and will NOT pick up new env
# vars for this singleton. You need to stop and re-run `streamlit run app.py`.
#
# To make this deterministic rather than depending on when some other code
# happens to call get_client() first, we construct the Langfuse client here,
# once, passing credentials explicitly instead of hoping ambient env state is
# correct when the singleton is first touched.

_LANGFUSE_STATE = {"client": None, "checked": False, "traced": False, "reason": None}


def _init_langfuse():
    """Build (and auth-check) the Langfuse client once. Cheap on repeat calls."""
    if _LANGFUSE_STATE["checked"]:
        return _LANGFUSE_STATE["traced"]

    _LANGFUSE_STATE["checked"] = True
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")

    if not (pub and sec):
        _LANGFUSE_STATE["reason"] = "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"
        return False

    debug = os.environ.get("LANGFUSE_DEBUG", "").lower() == "true"

    try:
        from langfuse import Langfuse
        # debug=True (or LANGFUSE_DEBUG=True in .env) turns on the SDK's own
        # verbose logging per the official troubleshooting docs - the most
        # direct way to see what auth_check() actually sent/received if the
        # short reason string below is not enough.
        client = Langfuse(public_key=pub, secret_key=sec, base_url=base_url, debug=debug)
    except Exception as exc:
        _LANGFUSE_STATE["reason"] = f"client construction failed - {type(exc).__name__}: {str(exc)[:200]}"
        return False

    try:
        # auth_check() is a real blocking network call. Worth it exactly once
        # at startup so a bad key or a blocked network path fails loudly here
        # instead of silently dropping every trace for the rest of the process.
        if client.auth_check():
            _LANGFUSE_STATE["client"] = client
            _LANGFUSE_STATE["traced"] = True
            return True
        _LANGFUSE_STATE["reason"] = (
            f"credentials rejected by {base_url or 'https://cloud.langfuse.com'} "
            "(auth_check returned false — check the key pair matches this base_url's region)"
        )
        return False
    except Exception as exc:
        # Distinguish "reached the server, credentials bad" from "never reached
        # the server" - these need different fixes and were previously reported
        # identically.
        name = type(exc).__name__
        text = str(exc)
        network_markers = ("ConnectError", "ConnectTimeout", "NameResolutionError",
                           "ConnectionError", "SSLError", "ReadTimeout", "ProxyError")
        if any(m in name or m in text for m in network_markers):
            _LANGFUSE_STATE["reason"] = (
                f"could not reach {base_url or 'https://cloud.langfuse.com'} - "
                f"{name}. This is a network/firewall/egress problem, not a credentials "
                "problem: check outbound access to that host from wherever this process runs "
                "(a container needs the host allowed in its egress policy)."
            )
        elif "401" in text or "403" in text or "Unauthorized" in text or "Forbidden" in text:
            _LANGFUSE_STATE["reason"] = (
                f"{base_url or 'https://cloud.langfuse.com'} rejected the key pair "
                f"({name}) - the project keys likely belong to a different region's project "
                "than the base_url points at."
            )
        else:
            _LANGFUSE_STATE["reason"] = f"{name}: {text[:200]}"
        return False


def langfuse_status(force: bool = False) -> dict:
    """For the sidebar: is tracing actually active, and if not, why."""
    if force:
        _LANGFUSE_STATE.update(client=None, checked=False, traced=False, reason=None)
    active = _init_langfuse()
    return {
        "active": active,
        "reason": _LANGFUSE_STATE["reason"],
        "base_url": os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
                    or "https://cloud.langfuse.com (default)",
        "debug_hint": (
            "Set LANGFUSE_DEBUG=true in .env and restart for verbose SDK logs in the "
            "terminal running `streamlit run` — the most direct way to see the raw "
            "request/response if the reason above is not specific enough."
        ),
    }


def _build_client():
    """Return (client, traced). Prefers the Langfuse drop-in wrapper."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, False

    if _init_langfuse():
        try:
            # The monkey-patch is global and keys off get_client()'s singleton,
            # which _init_langfuse() has now seeded with a verified client, so
            # this picks it up rather than racing to create its own.
            from langfuse.openai import openai as langfuse_openai
            return langfuse_openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL), True
        except Exception:
            pass  # fall through to an untraced client rather than losing the report

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL), False
    except Exception:
        return None, False


def _retry_after(exc, attempt: int) -> float:
    """Seconds to wait. Prefers the server's own hint over guessing."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(key)
        if raw:
            try:
                return min(float(str(raw).rstrip("s")), 30.0)
            except ValueError:
                continue
    return min(2.0 ** attempt, 30.0)


def generate_llm_report(evidence: dict, fmt: str = "json_compact",
                        model: str | None = None) -> dict:
    """Return the three-key report for one incident. Never raises.

    Environment:
        GROQ_API_KEY        required to make a real call
        GROQ_MODEL          optional, defaults to DEFAULT_MODEL
        LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
                            optional; all three present enables tracing
    """
    import time

    from llm_payload import serialise

    sanitize_env()
    client, traced = _build_client()
    if client is None:
        return fallback_report(evidence)

    messages = build_messages(serialise(evidence, fmt))
    model = model or resolve_model()
    if not model:
        return fallback_report(evidence, "no Groq model enabled for this project")

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=200,  # three short sentences; was 350
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    if traced:
        # Langfuse-only kwargs. A plain OpenAI client would reject these.
        kwargs["name"] = "incident-report"
        kwargs["metadata"] = {
            "driving_metric": evidence.get("driving_metric"),
            "window": evidence.get("window"),
            "has_culprit": evidence.get("has_culprit"),
        }

    last_error = "LLM unavailable"
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(**kwargs)
            _flush_if_traced(traced)
            report = parse_report(resp.choices[0].message.content)
            usage = getattr(resp, "usage", None)
            if usage:
                # Real numbers from Groq, not an estimate. This is what you
                # want to look at to see where the 1.5k tokens actually goes -
                # prompt_tokens vs completion_tokens tells you whether the fix
                # is a shorter system prompt or a lower max_tokens.
                report["_usage"] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "model": model,
                }
            return report

        except Exception as exc:
            name = type(exc).__name__
            text = str(exc)

            # Some models reject JSON mode. Drop it once and retry rather than
            # losing the report over a formatting flag.
            if "response_format" in kwargs and (
                "response_format" in text or "json_object" in text
            ):
                kwargs.pop("response_format")
                continue

            lowered = text.lower()
            if any(marker in lowered for marker in _BLOCKED_MARKERS):
                # Blocked at project level. Retrying the same ID is pointless,
                # but another enabled model may work — ask and switch once.
                alternative = resolve_model(force=True, exclude=model)
                if alternative and alternative != model and attempt < MAX_ATTEMPTS - 1:
                    model = alternative
                    kwargs["model"] = alternative
                    continue
                last_error = (
                    f"no usable Groq model — '{model}' is blocked for this project and "
                    "no enabled alternative was found. Enable one under "
                    "console.groq.com project limits"
                )
                break

            if "RateLimit" in name or "429" in text:
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(_retry_after(exc, attempt))
                    continue
                last_error = "rate limited"
                break

            if isinstance(exc, (ValueError, json.JSONDecodeError)):
                last_error = "unparseable model reply"
                break

            last_error = f"{name}"
            break

    return fallback_report(evidence, last_error)


def _flush_if_traced(traced: bool) -> None:
    """Force-send the buffered span for this call.

    The OTel BatchSpanProcessor flushes on its own schedule (a few seconds),
    which is fine for a long-running server but makes interactive debugging
    feel broken — you click, look at the Langfuse UI, see nothing, and
    conclude tracing is dead when it is just late. Flushing per-call trades a
    little latency for "the trace is there when you go look."
    """
    if not traced:
        return
    try:
        _LANGFUSE_STATE["client"].flush()
    except Exception:
        pass


def flush_traces() -> None:
    """Force-send buffered Langfuse events.

    The SDK batches in a background thread, which is fine under a long-running
    Streamlit server. Call this if traces are not appearing, or before exit.
    """
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass


# Backward-compatible shim: the previous entry point took a raw verdict and
# returned a string. Anything still calling it keeps working.
def generate_llm_diagnosis(verdict: dict) -> str:
    from llm_payload import build_evidence
    report = generate_llm_report(build_evidence(verdict))
    return " ".join(report[k] for k in REPORT_KEYS)