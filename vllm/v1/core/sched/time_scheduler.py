# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TokenCake Time Scheduler.

Decides whether to offload KV cache for stalled requests,
predicts function call durations via EWMA, and triggers
predictive uploads before function calls complete.
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.request import OffloadStatus

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class FCTypeStats:
    """Historical statistics for a given fc_type."""
    ewma_duration: float = 0.0
    count: int = 0
    alpha: float = 1.0       # Developer-estimate weight (decays over time)
    last_duration: float = 0.0
    max_observed: float = 0.0


# Cold-start defaults by fc_type base category
FC_TYPE_COLD_START: dict[str, float] = {
    "web_search": 3.0,
    "local_search": 1.0,
    "user_interaction": 60.0,
    "doc_processing": 5.0,
    "code_execution": 10.0,
    "citation_verify": 30.0,
    "default": 5.0,
}


@dataclass
class TimeSchedulerConfig:
    """Time Scheduler configuration parameters."""

    offload_enabled: bool = True

    # EWMA parameters
    ewma_beta: float = 0.3
    alpha_decay_rate: float = 0.1
    alpha_min: float = 0.2
    outlier_factor: float = 3.0

    # Offload decision parameters
    safety_margin: float = 1.5
    min_offload_window: float = 0.5  # seconds

    # Upload parameters
    upload_buffer_time: float = 0.1  # seconds

    # Transfer time model (hardware-dependent, needs calibration)
    offload_time_per_block: float = 0.001  # 1ms per block (HBM→Host)
    upload_time_per_block: float = 0.001   # 1ms per block (Host→HBM)

    # Gradual GPU Block Reservation
    gradual_reservation_enabled: bool = True
    reservation_cycles: int = 5


class TimeScheduler:
    """TokenCake Time Scheduler.

    Responsibilities:
    1. should_offload() — decide whether to offload a STALLED request's KV
    2. predict_fc_duration() — predict function call remaining time
    3. check_predictive_uploads() — check if upload should be triggered
    4. update_prediction() — update EWMA model on call_finish
    """

    def __init__(self, config: TimeSchedulerConfig | None = None):
        self.config = config or TimeSchedulerConfig()
        self.fc_stats: dict[str, FCTypeStats] = defaultdict(FCTypeStats)
        self.offload_time_per_block = self.config.offload_time_per_block
        self.upload_time_per_block = self.config.upload_time_per_block

    # ================================================================
    # 1. Offload Decision
    # ================================================================

    def should_offload(self, request: "Request",
                       num_kv_blocks: int = 0) -> bool:
        """Decide whether to offload a STALLED request's KV Cache.

        Returns True if the function call is predicted to last long enough
        that the scheduling benefit outweighs transfer overhead.
        """
        if not self.config.offload_enabled:
            return False

        T_transfer = self.calculate_transfer_time(num_kv_blocks)
        T_fc = self.predict_fc_duration(
            fc_type=request.fc_type,
            predict_time=request.fc_predict_time,
        )

        # Guard: function call too short to justify offload
        if T_fc <= T_transfer * self.config.safety_margin:
            return False

        T_window = T_fc - T_transfer
        if T_window < self.config.min_offload_window:
            return False

        return True

    # ================================================================
    # 2. Transfer Time Estimation
    # ================================================================

    def calculate_transfer_time(self, num_blocks: int) -> float:
        """Estimate round-trip transfer time (offload + upload)."""
        T_offload = num_blocks * self.offload_time_per_block
        T_upload = num_blocks * self.upload_time_per_block
        return T_offload + T_upload

    # ================================================================
    # 3. Function Call Duration Prediction
    # ================================================================

    def predict_fc_duration(
        self,
        fc_type: str | None,
        predict_time: float | None = None,
    ) -> float:
        """Predict total function call duration.

        Three-layer model:
          L0: Cold-start default (by fc_type base category)
          L1: Developer-supplied predict_time
          L2: EWMA historical average

        Fusion: t_final = alpha * t_dev + (1 - alpha) * t_hist
        alpha decays from 1.0 to alpha_min as observations accumulate.
        """
        if fc_type is None:
            return predict_time if predict_time is not None else \
                FC_TYPE_COLD_START["default"]

        stats = self.fc_stats.get(fc_type)
        fc_base = self._get_fc_base_type(fc_type)

        if stats is None or stats.count == 0:
            if predict_time is not None:
                return predict_time
            return FC_TYPE_COLD_START.get(fc_base,
                                          FC_TYPE_COLD_START["default"])

        t_hist = stats.ewma_duration
        if predict_time is not None:
            alpha = stats.alpha
            return alpha * predict_time + (1.0 - alpha) * t_hist
        return t_hist

    # ================================================================
    # 4. Predictive Upload Check
    # ================================================================

    def check_predictive_uploads(
        self,
        stalled_requests: list["Request"],
    ) -> list["Request"]:
        """Check which STALLED+OFFLOADED requests should trigger upload.

        Called each scheduling cycle. Returns requests needing upload.
        """
        to_upload: list[Request] = []
        now = time.monotonic()

        for req in stalled_requests:
            if req.offload_status != OffloadStatus.OFFLOADED:
                continue
            if req.fc_stall_start_time is None:
                continue

            elapsed = now - req.fc_stall_start_time
            predicted_total = self.predict_fc_duration(
                fc_type=req.fc_type,
                predict_time=req.fc_predict_time,
            )
            remaining = predicted_total - elapsed

            num_blocks = getattr(req, '_num_kv_blocks', 0)
            T_upload = num_blocks * self.upload_time_per_block
            buffer = self.config.upload_buffer_time

            if remaining <= T_upload + buffer:
                to_upload.append(req)

        return to_upload

    # ================================================================
    # 5. EWMA Update
    # ================================================================

    def update_prediction(
        self,
        fc_type: str | None,
        actual_duration: float,
    ) -> None:
        """Update EWMA model on call_finish."""
        if fc_type is None:
            return

        stats = self.fc_stats[fc_type]

        # Outlier filtering
        if (stats.count > 0 and
                actual_duration > stats.ewma_duration *
                self.config.outlier_factor):
            stats.max_observed = max(stats.max_observed, actual_duration)
            return

        if stats.count == 0:
            stats.ewma_duration = actual_duration
        else:
            beta = self.config.ewma_beta
            stats.ewma_duration = (beta * actual_duration +
                                   (1.0 - beta) * stats.ewma_duration)

        stats.last_duration = actual_duration
        stats.max_observed = max(stats.max_observed, actual_duration)
        stats.count += 1

        # Alpha decay
        stats.alpha = max(
            stats.alpha - self.config.alpha_decay_rate,
            self.config.alpha_min,
        )

    # ================================================================
    # 6. Gradual GPU Block Reservation
    # ================================================================

    def get_gradual_reservation_plan(
        self,
        num_blocks: int,
    ) -> list[int]:
        """Compute per-cycle block reservation plan for upload."""
        cycles = self.config.reservation_cycles
        blocks_per_cycle = math.ceil(num_blocks / cycles)
        plan: list[int] = []
        remaining = num_blocks
        for _ in range(cycles):
            n = min(blocks_per_cycle, remaining)
            plan.append(n)
            remaining -= n
            if remaining <= 0:
                break
        return plan

    # ================================================================
    # Internal Helpers
    # ================================================================

    @staticmethod
    def _get_fc_base_type(fc_type: str) -> str:
        """Extract base type from 'web_search:tavily' → 'web_search'."""
        return fc_type.split(":")[0] if ":" in fc_type else fc_type
