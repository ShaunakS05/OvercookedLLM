# utils/llm_async.py
"""
Fire-and-forget wrapper for OpenAI / Anthropic calls so the main game
thread stays responsive.

Usage:
    q = ask_llm_async(prompt, model="gpt-3.5-turbo", max_tokens=32, ...)
    ...
    try:
        txt = q.get_nowait()       # returns str  ▸ the LLM reply
    except queue.Empty:
        pass                       # still waiting
"""
from __future__ import annotations
import queue, concurrent.futures, atexit, traceback
import openai                         # or anthropic

# single worker keeps API requests serial (rate-limit friendly)
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=1)
atexit.register(_EXEC.shutdown, wait=False)

def _call_openai(prompt: str, **params) -> str:
    """Blocking HTTP call (runs in worker)."""
    resp = openai.ChatCompletion.create(        # ↙ pass any kwargs through
        messages=[{"role": "user", "content": prompt}], **params
    )
    return resp.choices[0].message["content"]

def ask_async(prompt: str, **params) -> queue.Queue:
    """Immediately returns a 1-slot Queue that will receive the reply/Exception."""
    out_q: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            out_q.put(_call_openai(prompt, **params))
        except Exception as exc:
            # surface the traceback in main thread
            out_q.put(RuntimeError(traceback.format_exc()))

    _EXEC.submit(_worker)
    return out_q
