# utils/llm_async.py
"""
Background-thread OpenAI helper that:
  • always enforces a 10-second ceiling (timeout/request_timeout)
  • removes any conflicting 'stream' kwarg before calling create()
  • prints debug info on start/end or error
"""

from __future__ import annotations
import queue, concurrent.futures, traceback, atexit, time
import openai

# ── single background worker ───────────────────────────────────────────
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=1)
atexit.register(_EXEC.shutdown, wait=False)

def _blocking_call(messages: list, **kw) -> str:
    # enforce timeouts for all openai-python versions
    kw.setdefault("timeout", 10)           # for ≥1.x
    kw.setdefault("request_timeout", 10)   # for 0.x

    # drop any incoming 'stream' to avoid multiple-values error
    kw.pop("stream", None)

    model = kw.get("model", "<unset>")
    print(f"[llm_async] ⇢ calling OpenAI (model={model})")
    t0 = time.time()
    try:
        resp = openai.ChatCompletion.create(
            messages=messages,
            stream=False,
            **kw
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[llm_async] ✗ OpenAI errored in {elapsed:.1f}s → {e}")
        raise
    elapsed = time.time() - t0
    print(f"[llm_async] ⇠ OpenAI returned in {elapsed:.1f}s")
    return resp.choices[0].message["content"]

def ask_async(messages: list, **kw) -> queue.Queue:
    """
    Fire-and-forget: returns an empty Queue immediately.
    Later, q.get_nowait() ⇒ str (reply) or RuntimeError (traceback).
    """
    q: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            reply = _blocking_call(messages, **kw)
            q.put(reply)
        except Exception:
            q.put(RuntimeError(traceback.format_exc()))

    _EXEC.submit(_worker)
    return q

def ask_llm_async(prompt: str, **kw) -> queue.Queue:
    """
    Legacy shim: wrap a single prompt string into ChatML format.
    """
    return ask_async([{"role": "user", "content": prompt}], **kw)
