"""Lightweight runtime error classification and event logging."""

from __future__ import annotations

import inspect
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


PROVIDER_TRANSPORT_MARKERS = (
    "remoteprotocolerror",
    "connecterror",
    "readerror",
    "readtimeout",
    "timeout_error",
    "apitimerouterror",
    "server disconnected",
    "peer closed connection",
    "incomplete chunked read",
    "connection reset",
    "connection aborted",
    "network is unreachable",
    "temporary failure in name resolution",
    "agent session failed",
)

PHASE_FUNCTIONS = (
    "propose_regions",
    "verify_region_crop",
    "refine_click_in_region",
    "verify_candidate",
    "rerank_candidates",
    "_refine_direct_on_crop",
    "screenspot_locate",
    "screenspot_crop_first_loop",
    "screenspot_active_loop",
    "refine_zoom_point",
    "locate_target",
    "find_target_in_known",
    "plan_next_action",
    "verify_step",
    "conclusion",
)


def _text_from(exc: BaseException | None = None, traceback_text: str | None = None) -> str:
    parts = []
    if exc is not None:
        parts.append(exc.__class__.__name__)
        parts.append(str(exc))
    if traceback_text:
        parts.append(traceback_text)
    return "\n".join(parts)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(marker in low for marker in markers)


def infer_phase_from_text(traceback_text: str | None) -> str | None:
    if not traceback_text:
        return None
    for fn_name in PHASE_FUNCTIONS:
        if f"in {fn_name}" in traceback_text:
            return fn_name
    return None


def infer_phase_from_stack() -> str | None:
    for frame in inspect.stack()[2:24]:
        if frame.function in PHASE_FUNCTIONS:
            return frame.function
    return None


def classify_exception(
    exc: BaseException | None = None,
    traceback_text: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    text = _text_from(exc, traceback_text)
    low = text.lower()
    has_provider_transport = _has_any(text, PROVIDER_TRANSPORT_MARKERS)

    # Prefer the provider's OWN structured reason when present —
    # OpenProgram's runtime.exec() raises a typed
    # ``LLMError(reason=ErrorReason.X, retryable=...)``, which is far more
    # reliable than grepping the message. (A SampleTimeoutError carries no
    # ``.reason`` and falls through to the text rules below.)
    reason = getattr(exc, "reason", None)
    reason_val = getattr(reason, "value", reason)  # ErrorReason enum → str
    reason_map = {
        "auth": ("provider_auth", False),
        "authz": ("provider_auth", False),
        "rate_limit": ("provider_rate_limit", True),
        "provider": ("provider_transport", True),
        "transport": ("provider_transport", True),
        "timeout": ("provider_timeout", bool(getattr(exc, "retryable", False))),
        "context": ("provider_invalid_request", False),
        "policy": ("provider_invalid_request", False),
        "invalid": ("provider_invalid_request", False),
    }
    if isinstance(reason_val, str) and reason_val in reason_map:
        category, retryable = reason_map[reason_val]
        return {
            "category": category,
            "retryable": retryable,
            "phase": phase or infer_phase_from_text(traceback_text),
        }

    if "sampletimeouterror" in low or "sample exceeded watchdog timeout" in low:
        category = "provider_timeout" if has_provider_transport else "sample_timeout"
        retryable = has_provider_transport
    elif "http 429" in low or "rate_limit" in low:
        category = "provider_rate_limit"
        retryable = True
    elif ("http 401" in low or "http 403" in low or "unauthorized" in low
          or "invalid api key" in low or "invalid_api_key" in low
          or "reason=auth" in low or "login expired" in low):
        category = "provider_auth"
        retryable = False
    elif has_provider_transport or "http 500" in low or "http 502" in low or "http 503" in low or "http 504" in low:
        category = "provider_transport"
        retryable = True
    elif "http 400" in low or "invalid_request" in low or "invalid image" in low:
        category = "provider_invalid_request"
        retryable = False
    elif "json" in low or "parse" in low:
        category = "model_parse"
        retryable = False
    else:
        category = "runtime_error"
        retryable = False

    inferred_phase = phase or infer_phase_from_text(traceback_text)
    return {
        "category": category,
        "retryable": retryable,
        "phase": inferred_phase,
    }


def reraise_if_fatal(exc: BaseException) -> None:
    """Re-raise provider errors a broad ``except Exception`` must NOT swallow.

    The locator wraps ``runtime.exec(...) + parse_json(...)`` in
    ``except Exception`` to recover from malformed model output. But that
    same net also catches OpenProgram's structured ``LLMError`` (auth,
    timeout, transport) and turns a real provider failure into a silent
    "uncertain" fallback — so the runner never sees the auth error and
    keeps burning samples on a dead credential. Call this first inside
    those handlers: it re-raises the errors the runner must classify /
    stop on, while genuine parse errors fall through to the fallback.

    ``ExecInterrupt`` is a ``BaseException`` so ``except Exception`` already
    lets it through; we still re-raise it explicitly in case a caller used
    a broader ``except BaseException``.
    """
    try:
        from openprogram.providers.utils.errors import LLMError, ExecInterrupt
    except Exception:
        return
    if isinstance(exc, (LLMError, ExecInterrupt)):
        raise exc


def _summarize_content(content: Any) -> dict[str, Any]:
    if not isinstance(content, list):
        return {"kind": type(content).__name__}
    text_chars = 0
    image_count = 0
    image_paths: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text_chars += len(str(item.get("text") or ""))
        elif item.get("type") == "image":
            image_count += 1
            path = item.get("path")
            if path:
                image_paths.append(str(path))
    return {
        "items": len(content),
        "text_chars": text_chars,
        "image_count": image_count,
        "image_paths": image_paths[:8],
    }


def record_runtime_error(
    exc: BaseException,
    *,
    phase: str | None = None,
    content: Any = None,
) -> None:
    path_value = os.environ.get("GUI_HARNESS_ERROR_EVENTS")
    if not path_value:
        return

    tb = traceback.format_exc(limit=20)
    classification = classify_exception(exc, traceback_text=tb, phase=phase or infer_phase_from_stack())
    event = {
        "ts": time.time(),
        "pid": os.getpid(),
        "phase": classification.get("phase"),
        "category": classification["category"],
        "retryable": classification["retryable"],
        "error_type": exc.__class__.__name__,
        "message": str(exc)[:2000],
        "content": _summarize_content(content),
    }
    try:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Error monitoring must never become a second failure path.
        return
