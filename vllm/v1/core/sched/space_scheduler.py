# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TokenCake Space Scheduler.

Computes hybrid priority (DAG static + runtime dynamic) and manages
KV cache memory partitioning between critical and non-critical agents.
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class AgentTypeStats:
    """Runtime statistics for an agent_type."""
    static_priority: float = 0.0
    active_request_count: int = 0
    avg_block_usage: float = 0.0
    total_block_usage: int = 0
    hybrid_score: float = 0.0


@dataclass
class SpaceSchedulerConfig:
    """Space Scheduler configuration."""

    priority_enabled: bool = True

    # Priority parameters
    w_static: float = 10.0

    # Memory partition parameters
    initial_reserve_ratio: float = 0.1
    min_reserve_ratio: float = 0.0
    max_reserve_ratio: float = 0.4
    gpu_usage_high: float = 0.85
    gpu_usage_low: float = 0.60
    adjustment_step: float = 0.05

    # Critical agent selection
    critical_ratio: float = 0.5

    # Update frequency
    update_interval_steps: int = 10


class SpaceScheduler:
    """TokenCake Space Scheduler.

    Responsibilities:
    1. Compute hybrid priority, inject into Request.priority
    2. Dynamically manage KV cache memory partitions (shared/reserved)
    3. Periodically select critical agents for block reservation
    """

    def __init__(self, config: SpaceSchedulerConfig | None = None):
        self.config = config or SpaceSchedulerConfig()
        self.agent_stats: dict[str, AgentTypeStats] = defaultdict(
            AgentTypeStats)

        # Memory partition state
        self.total_reserve_ratio: float = self.config.initial_reserve_ratio
        self.reserve_allocations: dict[str, int] = {}
        self.critical_agents: set[str] = set()

    # ================================================================
    # 1. Hybrid Priority Computation
    # ================================================================

    def compute_hybrid_priority(self, request: "Request") -> float:
        """Compute hybrid priority score for a request.

        Returns a value where higher = more critical.
        This is then mapped to vLLM's priority (lower = more priority).
        """
        static_p = request.static_priority
        dynamic_p = self._compute_dynamic_priority(request)
        return static_p * self.config.w_static + dynamic_p

    def _compute_dynamic_priority(self, request: "Request") -> float:
        """dynamic_priority = time_wait * log(tokens_req / time_wait)"""
        now = time.monotonic()
        time_wait = max(now - request.arrival_time, 0.01)
        tokens_req = max(
            request.num_prompt_tokens + request.max_tokens, 1)

        ratio = tokens_req / time_wait
        if ratio <= 0:
            return time_wait
        return time_wait * math.log(ratio)

    def update_request_priority(self, request: "Request") -> None:
        """Update request's priority field based on hybrid score.

        Called when adding a request and during periodic refresh.
        """
        if not self.config.priority_enabled:
            return
        if request.agent_type is not None:
            hybrid = self.compute_hybrid_priority(request)
            # vLLM: lower priority value = higher priority
            request.priority = int(-hybrid * 1000)

    # ================================================================
    # 2. Memory Dynamic Partitioning
    # ================================================================

    def update_memory_reservations(
        self,
        total_gpu_blocks: int,
        used_gpu_blocks: int,
    ) -> dict[str, int]:
        """Periodically adjust memory partitions.

        Phase 1: Adjust reserved pool total size based on GPU utilization.
        Phase 2: Distribute reserved pool among critical agents.

        Returns: {agent_type: num_reserved_blocks}
        """
        # Phase 1: Adjust reserved pool size
        usage_ratio = used_gpu_blocks / max(total_gpu_blocks, 1)

        if usage_ratio >= self.config.gpu_usage_high:
            self.total_reserve_ratio = min(
                self.total_reserve_ratio + self.config.adjustment_step,
                self.config.max_reserve_ratio,
            )
        elif usage_ratio <= self.config.gpu_usage_low:
            self.total_reserve_ratio = max(
                self.total_reserve_ratio - self.config.adjustment_step,
                self.config.min_reserve_ratio,
            )

        R_total = int(total_gpu_blocks * self.total_reserve_ratio)

        # Phase 2: Distribute among critical agents
        if not self.critical_agents:
            self.reserve_allocations = {}
            return {}

        S_total = sum(
            self.agent_stats[a].hybrid_score
            for a in self.critical_agents
        )
        if S_total <= 0:
            per_agent = R_total // len(self.critical_agents)
            self.reserve_allocations = {
                a: per_agent for a in self.critical_agents}
            return self.reserve_allocations

        reserve_num: dict[str, int] = {}
        for agent_type in self.critical_agents:
            stats = self.agent_stats[agent_type]
            mem_ratio = stats.avg_block_usage / max(total_gpu_blocks, 1)
            priority_ratio = stats.hybrid_score / S_total
            final_ratio = (mem_ratio + priority_ratio) / 2.0
            reserve_num[agent_type] = int(final_ratio * R_total)

        self.reserve_allocations = reserve_num
        return reserve_num

    # ================================================================
    # 3. Critical Agent Selection
    # ================================================================

    def select_critical_agents(self) -> set[str]:
        """Select current critical agent set (top-K by hybrid score)."""
        scored = []
        for agent_type, stats in self.agent_stats.items():
            if stats.active_request_count > 0:
                scored.append((agent_type, stats.hybrid_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        n_critical = max(1, int(len(scored) * self.config.critical_ratio))
        self.critical_agents = {a for a, _ in scored[:n_critical]}
        return self.critical_agents

    # ================================================================
    # 4. Statistics Updates
    # ================================================================

    def register_agent_meta(
        self,
        agent_type: str,
        static_priority: float,
    ) -> None:
        """Register agent metadata."""
        stats = self.agent_stats[agent_type]
        stats.static_priority = max(stats.static_priority, static_priority)

    def update_agent_stats(
        self,
        agent_type: str,
        block_count: int,
        request_count: int,
    ) -> None:
        """Update agent_type runtime statistics."""
        stats = self.agent_stats[agent_type]
        stats.active_request_count = request_count
        stats.total_block_usage = block_count

        beta = 0.3
        stats.avg_block_usage = (
            beta * block_count + (1 - beta) * stats.avg_block_usage
        )
        stats.hybrid_score = stats.static_priority
