"""Dependency-free progress reporting for long verifier loops.

The certify stage spends almost all of its wall time inside chunked CROWN
passes (``crown_bounds_chunked``), which can run tens of minutes with no
output.  When stdout is a pipe/file (e.g. a background run) Python block-
buffers it, so nothing is visible until the stage exits.  ``Progress`` gives a
live percentage + ETA that works in *both* settings:

  * interactive TTY  -> a single ``\\r``-updated bar (no scroll spam);
  * piped / file     -> throttled, newline-terminated, ``flush``-ed lines.

It is pure observability: it touches no bound, never enters the soundness
argument, and adds at most one cheap ``time.time()`` per chunk.
"""
from __future__ import annotations

import sys
import time


class Progress:
    """Throttled progress meter over ``total`` units of work."""

    def __init__(self, total: int, desc: str = "", every_s: float = 5.0,
                 width: int = 26, stream=None) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.every_s = every_s
        self.width = width
        self.stream = stream if stream is not None else sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.t0 = time.time()
        self._last_emit = 0.0
        self.n = 0

    # ------------------------------------------------------------------ #
    def update(self, n: int) -> None:
        """Report that ``n`` of ``total`` units are complete."""
        self.n = n
        now = time.time()
        if now - self._last_emit >= self.every_s and n < self.total:
            self._emit(now, final=False)
            self._last_emit = now

    def done(self) -> None:
        self._emit(time.time(), final=True)

    # ------------------------------------------------------------------ #
    def _emit(self, now: float, final: bool) -> None:
        frac = min(self.n / self.total, 1.0)
        el = now - self.t0
        eta = el * (1.0 - frac) / frac if frac > 1e-9 else float("inf")
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        line = (f"    [{self.desc}] [{bar}] {100 * frac:5.1f}%  "
                f"{self.n}/{self.total}  {el:5.0f}s"
                f"{'' if final else f'  ETA {eta:4.0f}s'}")
        if self.is_tty:
            self.stream.write("\r" + line + ("\n" if final else ""))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
