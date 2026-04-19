# SPDX-License-Identifier: Apache-2.0
"""
mflux callback that reports denoising step progress to ImageProgressTracker.

Implements mflux's duck-typed callback protocols (BeforeLoopCallback,
InLoopCallback, AfterLoopCallback) to track per-step progress during image
generation. Registered via model.callbacks.register() before generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

    from ..image_progress import ImageProgressTracker

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 50


class ImageProgressCallback:
    """mflux callback that reports denoising step progress.

    Implements duck-typed callback interfaces detected by CallbackRegistry:
    - call_before_loop: registers request in tracker
    - call_in_loop: updates step progress per denoising step
    - call_after_loop: removes request from tracker on completion
    """

    def __init__(
        self,
        request_id: str,
        model_id: str,
        tracker: ImageProgressTracker,
    ) -> None:
        self.request_id = request_id
        self.model_id = model_id
        self.tracker = tracker

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: object,
        **kwargs,
    ) -> None:
        total_steps = getattr(config, "num_inference_steps", 0)
        self.tracker.update(
            request_id=self.request_id,
            model_id=self.model_id,
            step=0,
            total_steps=total_steps,
            seed=seed,
            prompt=prompt[:MAX_PROMPT_LENGTH],
        )

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: object,
        time_steps: object,
        **kwargs,
    ) -> None:
        total_steps = getattr(config, "num_inference_steps", 0)
        self.tracker.update(
            request_id=self.request_id,
            model_id=self.model_id,
            step=t + 1,
            total_steps=total_steps,
            seed=seed,
            prompt=prompt[:MAX_PROMPT_LENGTH],
        )

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: object,
        **kwargs,
    ) -> None:
        self.tracker.remove(self.request_id)

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: object,
        time_steps: object,
        **kwargs,
    ) -> None:
        """Handle interrupted generation: remove from tracker.

        Note: Callback cleanup from model.callbacks is handled in ImageEngine's
        finally block, not here, to avoid modifying callbacks during interrupt.
        """
        self.tracker.remove(self.request_id)
