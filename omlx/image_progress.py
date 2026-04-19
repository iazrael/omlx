# SPDX-License-Identifier: Apache-2.0
"""
Lightweight image generation progress tracker for dashboard display.

Updated by ImageProgressCallback on each denoising step (CPU counters only,
zero GPU overhead). Read by admin stats API to show per-request step progress
in the Active Models card.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class ImageProgressTracker:
    """Thread-safe tracker for per-request image generation progress.

    Each entry stores (step, total_steps, model_id, seed, prompt, timing) for
    a request that is currently generating an image.  Entries are removed
    explicitly when generation completes or in a finally block.

    Performance: ~50ns lock acquire/release + O(1) dict write per step.
    Called once per denoising step (typically 4-50 steps total).
    """

    def __init__(self) -> None:
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def update(
        self,
        request_id: str,
        model_id: str,
        step: int,
        total_steps: int,
        seed: int,
        prompt: str,
    ) -> None:
        """Update image generation progress for a request."""
        with self._lock:
            now = time.monotonic()
            prev = self._progress.get(request_id)

            if prev is None:
                # New request: initialize with zero speed
                start_time = now
                speed = 0.0
            else:
                # Existing request: calculate step speed
                dt = now - prev["last_time"]
                dsteps = step - prev["step"]
                speed = dsteps / dt if dt > 0 and dsteps > 0 else prev.get("speed", 0.0)
                start_time = prev["start_time"]

            self._progress[request_id] = {
                "step": step,
                "total_steps": total_steps,
                "model_id": model_id,
                "seed": seed,
                "prompt": prompt,
                "start_time": start_time,
                "last_time": now,
                "speed": speed,
            }

    def remove(self, request_id: str) -> None:
        """Remove a request (on completion, abort, or error)."""
        with self._lock:
            self._progress.pop(request_id, None)

    def get_model_progress(self, model_id: str) -> List[Dict[str, Any]]:
        """Return list of generating requests for a given model."""
        with self._lock:
            results = []
            for rid, entry in self._progress.items():
                if entry["model_id"] != model_id:
                    continue
                elapsed = entry["last_time"] - entry["start_time"]
                speed = entry.get("speed", 0.0)
                results.append({
                    "request_id": rid,
                    "step": entry["step"],
                    "total_steps": entry["total_steps"],
                    "seed": entry["seed"],
                    "prompt": entry["prompt"],
                    "elapsed": round(elapsed, 1),
                    "speed": round(speed, 2),
                })
            return results

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._progress.clear()


# Module-level singleton, lazily created.
_tracker: Optional[ImageProgressTracker] = None
_tracker_lock = threading.Lock()


def get_image_progress_tracker() -> ImageProgressTracker:
    """Get or create the global ImageProgressTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ImageProgressTracker()
    return _tracker
