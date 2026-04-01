# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for TokenCake components."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.offload_router import (
    OffloadRouter,
    OffloadRouterConfig,
    OffloadTarget,
)
from vllm.v1.core.sched.space_scheduler import (
    SpaceScheduler,
    SpaceSchedulerConfig,
)
from vllm.v1.core.sched.time_scheduler import (
    TimeScheduler,
    TimeSchedulerConfig,
)
from vllm.v1.request import OffloadStatus, RequestStatus


def _make_request(
    request_id: str = "req-1",
    fc_type: str | None = None,
    fc_predict_time: float | None = None,
    fc_stall_start_time: float | None = None,
    offload_status: OffloadStatus = OffloadStatus.NONE,
    agent_type: str | None = None,
    static_priority: float = 0.0,
    arrival_time: float | None = None,
    num_prompt_tokens: int = 100,
    max_tokens: int = 200,
) -> SimpleNamespace:
    """Create a mock request for testing."""
    return SimpleNamespace(
        request_id=request_id,
        fc_type=fc_type,
        fc_predict_time=fc_predict_time,
        fc_stall_start_time=fc_stall_start_time,
        offload_status=offload_status,
        agent_type=agent_type,
        static_priority=static_priority,
        arrival_time=arrival_time or time.monotonic(),
        num_prompt_tokens=num_prompt_tokens,
        max_tokens=max_tokens,
        priority=0,
        _num_kv_blocks=10,
        status=RequestStatus.RUNNING,
    )


# ============================================================
# TimeScheduler Tests
# ============================================================


class TestTimeScheduler:
    def test_predict_fc_duration_cold_start_with_predict_time(self):
        ts = TimeScheduler()
        result = ts.predict_fc_duration("web_search:tavily", predict_time=5.0)
        assert result == 5.0

    def test_predict_fc_duration_cold_start_default(self):
        ts = TimeScheduler()
        result = ts.predict_fc_duration("web_search:tavily")
        assert result == 3.0  # web_search default

    def test_predict_fc_duration_cold_start_unknown_type(self):
        ts = TimeScheduler()
        result = ts.predict_fc_duration("unknown_type")
        assert result == 5.0  # "default" fallback

    def test_predict_fc_duration_none_type(self):
        ts = TimeScheduler()
        assert ts.predict_fc_duration(None) == 5.0
        assert ts.predict_fc_duration(None, predict_time=3.0) == 3.0

    def test_ewma_update(self):
        ts = TimeScheduler(TimeSchedulerConfig(ewma_beta=0.3))
        ts.update_prediction("web_search:tavily", 2.0)
        assert ts.fc_stats["web_search:tavily"].ewma_duration == 2.0
        assert ts.fc_stats["web_search:tavily"].count == 1

        ts.update_prediction("web_search:tavily", 4.0)
        # EWMA: 0.3 * 4.0 + 0.7 * 2.0 = 2.6
        assert abs(ts.fc_stats["web_search:tavily"].ewma_duration - 2.6) < 0.01

    def test_ewma_alpha_decay(self):
        ts = TimeScheduler(TimeSchedulerConfig(
            alpha_decay_rate=0.1, alpha_min=0.2))
        for _ in range(10):
            ts.update_prediction("test_type", 3.0)
        stats = ts.fc_stats["test_type"]
        assert stats.alpha == 0.2  # Decayed to min

    def test_ewma_outlier_filtering(self):
        ts = TimeScheduler(TimeSchedulerConfig(outlier_factor=3.0))
        ts.update_prediction("test", 2.0)
        initial_ewma = ts.fc_stats["test"].ewma_duration

        # Outlier: 10x the mean
        ts.update_prediction("test", 20.0)
        assert ts.fc_stats["test"].ewma_duration == initial_ewma
        assert ts.fc_stats["test"].max_observed == 20.0

    def test_predict_with_history_and_dev_estimate(self):
        ts = TimeScheduler()
        # Build up history
        for _ in range(5):
            ts.update_prediction("web_search:google", 3.0)

        stats = ts.fc_stats["web_search:google"]
        # alpha should have decayed: 1.0 - 5*0.1 = 0.5
        assert abs(stats.alpha - 0.5) < 0.01

        # Predict with dev estimate
        result = ts.predict_fc_duration("web_search:google",
                                         predict_time=5.0)
        # 0.5 * 5.0 + 0.5 * 3.0 = 4.0
        assert abs(result - 4.0) < 0.1

    def test_should_offload_short_call(self):
        ts = TimeScheduler(TimeSchedulerConfig(
            safety_margin=1.5,
            min_offload_window=0.5,
            offload_time_per_block=0.001,
            upload_time_per_block=0.001,
        ))
        req = _make_request(fc_type="local_search", fc_predict_time=0.1)
        # 0.1s call, 10 blocks * 0.002 = 0.02s transfer
        # 0.1 <= 0.02 * 1.5 = 0.03? No, but window = 0.1-0.02 = 0.08 < 0.5
        assert ts.should_offload(req, num_kv_blocks=10) is False

    def test_should_offload_long_call(self):
        ts = TimeScheduler(TimeSchedulerConfig(
            safety_margin=1.5,
            min_offload_window=0.5,
        ))
        req = _make_request(fc_type="web_search:tavily",
                            fc_predict_time=10.0)
        assert ts.should_offload(req, num_kv_blocks=10) is True

    def test_should_offload_disabled(self):
        ts = TimeScheduler(TimeSchedulerConfig(offload_enabled=False))
        req = _make_request(fc_type="web_search", fc_predict_time=10.0)
        assert ts.should_offload(req, num_kv_blocks=10) is False

    def test_check_predictive_uploads(self):
        ts = TimeScheduler(TimeSchedulerConfig(
            upload_buffer_time=0.1,
            upload_time_per_block=0.001,
        ))
        # Update prediction so we have history
        ts.update_prediction("web_search", 2.0)

        # Create a request that was stalled 1.9s ago (close to 2.0s predicted)
        req = _make_request(
            fc_type="web_search",
            offload_status=OffloadStatus.OFFLOADED,
            fc_stall_start_time=time.monotonic() - 1.9,
        )

        result = ts.check_predictive_uploads([req])
        assert len(result) == 1
        assert result[0] is req

    def test_check_predictive_uploads_too_early(self):
        ts = TimeScheduler(TimeSchedulerConfig(
            upload_buffer_time=0.1,
            upload_time_per_block=0.001,
        ))
        ts.update_prediction("code_execution", 10.0)

        # Stalled only 1s ago, predicted 10s
        req = _make_request(
            fc_type="code_execution",
            offload_status=OffloadStatus.OFFLOADED,
            fc_stall_start_time=time.monotonic() - 1.0,
        )

        result = ts.check_predictive_uploads([req])
        assert len(result) == 0

    def test_gradual_reservation_plan(self):
        ts = TimeScheduler(TimeSchedulerConfig(reservation_cycles=5))
        plan = ts.get_gradual_reservation_plan(num_blocks=23)
        assert sum(plan) == 23
        assert len(plan) == 5

    def test_get_fc_base_type(self):
        assert TimeScheduler._get_fc_base_type("web_search:tavily") == \
            "web_search"
        assert TimeScheduler._get_fc_base_type("code_execution") == \
            "code_execution"


# ============================================================
# OffloadRouter Tests
# ============================================================


class TestOffloadRouter:
    def test_short_call_host_ram(self):
        router = OffloadRouter()
        router.initialize(has_host_offload=True, has_ucm_connector=True)
        req = _make_request()
        target = router.select_target(req, predict_time=2.0)
        assert target == OffloadTarget.HOST_RAM

    def test_medium_call_ucm_local(self):
        router = OffloadRouter()
        router.initialize(has_host_offload=True, has_ucm_connector=True)
        req = _make_request()
        target = router.select_target(req, predict_time=30.0)
        assert target == OffloadTarget.UCM_LOCAL

    def test_long_call_ucm_remote(self):
        router = OffloadRouter()
        router.initialize(has_host_offload=True, has_ucm_connector=True)
        req = _make_request()
        target = router.select_target(req, predict_time=120.0)
        assert target == OffloadTarget.UCM_REMOTE

    def test_no_backends_returns_none(self):
        router = OffloadRouter()
        router.initialize(has_host_offload=False, has_ucm_connector=False)
        req = _make_request()
        target = router.select_target(req, predict_time=5.0)
        assert target is None

    def test_fallback_host_when_no_ucm(self):
        router = OffloadRouter()
        router.initialize(has_host_offload=True, has_ucm_connector=False)
        req = _make_request()
        target = router.select_target(req, predict_time=30.0)
        assert target == OffloadTarget.HOST_RAM

    def test_fallback_ucm_when_no_host(self):
        router = OffloadRouter()
        router.initialize(has_host_offload=False, has_ucm_connector=True)
        req = _make_request()
        target = router.select_target(req, predict_time=2.0)
        assert target == OffloadTarget.UCM_LOCAL


# ============================================================
# SpaceScheduler Tests
# ============================================================


class TestSpaceScheduler:
    def test_hybrid_priority_with_static(self):
        ss = SpaceScheduler(SpaceSchedulerConfig(w_static=10.0))
        req = _make_request(agent_type="programmer", static_priority=0.9)
        hybrid = ss.compute_hybrid_priority(req)
        assert hybrid > 0  # Should be positive

    def test_update_request_priority(self):
        ss = SpaceScheduler()
        req = _make_request(agent_type="programmer", static_priority=0.9)
        old_priority = req.priority
        ss.update_request_priority(req)
        assert req.priority != old_priority

    def test_update_request_priority_no_agent_type(self):
        ss = SpaceScheduler()
        req = _make_request(agent_type=None)
        old_priority = req.priority
        ss.update_request_priority(req)
        assert req.priority == old_priority  # Not modified

    def test_update_request_priority_disabled(self):
        ss = SpaceScheduler(SpaceSchedulerConfig(priority_enabled=False))
        req = _make_request(agent_type="programmer", static_priority=0.9)
        old_priority = req.priority
        ss.update_request_priority(req)
        assert req.priority == old_priority

    def test_register_agent_meta(self):
        ss = SpaceScheduler()
        ss.register_agent_meta("programmer", 0.9)
        assert ss.agent_stats["programmer"].static_priority == 0.9

    def test_select_critical_agents(self):
        ss = SpaceScheduler(SpaceSchedulerConfig(critical_ratio=0.5))
        ss.agent_stats["prog"].static_priority = 0.9
        ss.agent_stats["prog"].hybrid_score = 0.9
        ss.agent_stats["prog"].active_request_count = 1
        ss.agent_stats["review"].static_priority = 0.3
        ss.agent_stats["review"].hybrid_score = 0.3
        ss.agent_stats["review"].active_request_count = 1

        critical = ss.select_critical_agents()
        assert "prog" in critical
        assert len(critical) == 1

    def test_memory_reservations_high_usage(self):
        ss = SpaceScheduler(SpaceSchedulerConfig(
            initial_reserve_ratio=0.1,
            adjustment_step=0.05,
            gpu_usage_high=0.85,
        ))
        ss.critical_agents = {"prog"}
        ss.agent_stats["prog"].hybrid_score = 1.0
        ss.agent_stats["prog"].avg_block_usage = 50.0

        result = ss.update_memory_reservations(
            total_gpu_blocks=1000,
            used_gpu_blocks=900,
        )
        # Reserve ratio should increase from 0.1 to 0.15
        assert abs(ss.total_reserve_ratio - 0.15) < 0.001


# ============================================================
# Request Status Tests
# ============================================================


class TestRequestStatus:
    def test_stalled_on_fc_not_finished(self):
        assert RequestStatus.is_finished(RequestStatus.STALLED_ON_FC) is False

    def test_stalled_on_fc_ordering(self):
        assert RequestStatus.RUNNING < RequestStatus.STALLED_ON_FC
        assert RequestStatus.STALLED_ON_FC < RequestStatus.PREEMPTED


class TestOffloadStatus:
    def test_status_values(self):
        assert OffloadStatus.NONE == 0
        assert OffloadStatus.OFFLOADING == 1
        assert OffloadStatus.OFFLOADED == 2
        assert OffloadStatus.UPLOADING == 3
        assert OffloadStatus.UPLOAD_COMPLETE == 4
