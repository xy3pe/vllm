# 04 — Space Scheduler 设计

> 基于 DAG 关键路径和运行时状态的混合优先级 + KV Cache 内存动态分区。

---

## 1. 组件定位

```
                 agent_meta 注册
                      │
                      ▼
            ┌────────────────────┐
            │  Space Scheduler   │
            │                    │
            │ hybrid_priority()  │──▶ 注入 PriorityRequestQueue
            │                    │
            │ update_memory_     │──▶ 调整 shared/reserved 分区
            │   reservations()   │
            │                    │
            │ select_critical_   │──▶ 确定哪些 agent_type 受保护
            │   agents()         │
            └────────────────────┘
```

Space Scheduler 解决**关键路径倒置**问题：非关键 agent 抢占关键 agent 的 KV Cache，
导致整个应用端到端延迟增加。

---

## 2. 混合优先级

### 2.1 公式

```
hybrid_priority = static_priority * w_static + dynamic_priority

其中:
  static_priority  — 上层应用传入，反映 DAG 关键路径权重（0~1）
  dynamic_priority — 运行时计算，平衡公平性和吞吐
  w_static         — 可配置的静态权重系数
```

### 2.2 动态优先级

```
dynamic_priority = time_wait * log(tokens_req / time_wait)

其中:
  time_wait  = 请求在系统中的等待时间（秒）
  tokens_req = 请求的总 token 数
```

- **线性项 (time_wait)**: 防止饥饿，等待越久优先级越高
- **对数项 (log(tokens_req/time_wait))**: 适度偏向短请求，提升吞吐

### 2.3 实现

```python
# vllm/v1/core/sched/space_scheduler.py (新建)

import math
import time
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AgentTypeStats:
    """某个 agent_type 的运行时统计。"""
    static_priority: float = 0.0       # 上层传入的 DAG 权重
    active_request_count: int = 0      # 当前活跃请求数
    avg_block_usage: float = 0.0       # 平均 block 占用
    total_block_usage: int = 0         # 当前总 block 占用
    hybrid_score: float = 0.0          # 最近计算的混合分数


class SpaceScheduler:
    """
    TokenCake Space Scheduler.

    职责:
    1. 计算混合优先级，注入 Request 的 priority 字段
    2. 动态管理 KV Cache 内存分区（shared pool / reserved pool）
    3. 周期性选择关键 agent，为其预留 blocks
    """

    def __init__(self, config: "SpaceSchedulerConfig"):
        self.config = config
        self.agent_stats: dict[str, AgentTypeStats] = defaultdict(AgentTypeStats)

        # 内存分区状态
        self.total_reserve_ratio: float = config.initial_reserve_ratio
        self.reserve_allocations: dict[str, int] = {}  # agent_type → reserved blocks
        self.critical_agents: set[str] = set()

    # ================================================================
    # 1. 混合优先级计算
    # ================================================================

    def compute_hybrid_priority(self, request: "Request") -> float:
        """
        计算请求的混合优先级分数。

        返回值越小 = 优先级越高（与 vLLM 的 Request.__lt__ 一致）。
        因此需要取负数。
        """
        static_p = request.static_priority  # 上层传入，0~1，越大越关键

        # 动态优先级
        dynamic_p = self._compute_dynamic_priority(request)

        # 混合（越大越关键）
        hybrid = static_p * self.config.w_static + dynamic_p

        # 转为 vLLM priority（越小越优先）
        # 使用负数映射：hybrid 越大 → priority 越小（越优先）
        return -hybrid

    def _compute_dynamic_priority(self, request: "Request") -> float:
        """
        dynamic_priority = time_wait * log(tokens_req / time_wait)
        """
        now = time.monotonic()
        time_wait = max(now - request.arrival_time, 0.01)  # 避免 0
        tokens_req = max(request.num_prompt_tokens + request.max_tokens, 1)

        ratio = tokens_req / time_wait
        if ratio <= 0:
            return time_wait  # fallback

        return time_wait * math.log(ratio)

    def update_request_priority(self, request: "Request"):
        """
        更新请求的 priority 字段。

        在 add_request() 时和周期性刷新时调用。
        """
        if not self.config.priority_enabled:
            return

        if request.agent_type is not None:
            request.priority = int(
                self.compute_hybrid_priority(request) * 1000
            )

    # ================================================================
    # 2. 内存动态分区 (Algorithm 2)
    # ================================================================

    def update_memory_reservations(
        self,
        total_gpu_blocks: int,
        used_gpu_blocks: int,
    ) -> dict[str, int]:
        """
        周期性调整内存分区。

        Phase 1: 根据 GPU 利用率调整 reserved pool 总大小
        Phase 2: 按优先级和用量分配 reserved pool 给各 critical agent

        返回: {agent_type: num_reserved_blocks}
        """
        # ---- Phase 1: 调整 reserved pool 总大小 ----
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

        # ---- Phase 2: 分配给各 critical agent ----
        if not self.critical_agents:
            self.reserve_allocations = {}
            return {}

        S_total = sum(
            self.agent_stats[a].hybrid_score
            for a in self.critical_agents
        )
        if S_total <= 0:
            # 均分
            per_agent = R_total // len(self.critical_agents)
            self.reserve_allocations = {a: per_agent for a in self.critical_agents}
            return self.reserve_allocations

        reserve_num = {}
        for agent_type in self.critical_agents:
            stats = self.agent_stats[agent_type]
            mem_ratio = stats.avg_block_usage / max(total_gpu_blocks, 1)
            priority_ratio = stats.hybrid_score / S_total

            final_ratio = (mem_ratio + priority_ratio) / 2.0
            reserve_num[agent_type] = int(final_ratio * R_total)

        self.reserve_allocations = reserve_num
        return reserve_num

    # ================================================================
    # 3. 关键 Agent 选择
    # ================================================================

    def select_critical_agents(self) -> set[str]:
        """
        选择当前关键 agent 集合。

        取混合分数最高的 top-K agent_type。
        """
        # 更新所有 agent 的混合分数
        scored = []
        for agent_type, stats in self.agent_stats.items():
            if stats.active_request_count > 0:
                scored.append((agent_type, stats.hybrid_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 取 top critical_ratio
        n_critical = max(1, int(len(scored) * self.config.critical_ratio))
        self.critical_agents = {a for a, _ in scored[:n_critical]}
        return self.critical_agents

    # ================================================================
    # 4. 统计更新
    # ================================================================

    def register_agent_meta(
        self,
        request_id: str,
        agent_type: str,
        static_priority: float,
    ):
        """注册 agent 元数据。"""
        stats = self.agent_stats[agent_type]
        stats.static_priority = max(stats.static_priority, static_priority)

    def update_agent_stats(
        self,
        agent_type: str,
        block_count: int,
        request_count: int,
    ):
        """更新 agent_type 的运行时统计。"""
        stats = self.agent_stats[agent_type]
        stats.active_request_count = request_count
        stats.total_block_usage = block_count

        # EWMA 更新 avg_block_usage
        beta = 0.3
        stats.avg_block_usage = (
            beta * block_count + (1 - beta) * stats.avg_block_usage
        )

        # 更新 hybrid_score（用 static_priority 代表该类型的整体分数）
        stats.hybrid_score = stats.static_priority
```

---

## 3. KVCacheManager 内存分区集成

### 3.1 Block Pool 分区

```python
# vllm/v1/core/kv_cache_manager.py — 扩展

class KVCacheManager:
    def __init__(self, ...):
        # ... 现有初始化 ...

        # TokenCake: 内存分区
        self.reserved_blocks: dict[str, set[int]] = {}  # agent_type → block_ids
        self.shared_blocks: set[int] = set()             # 共享 pool 的 block_ids

    def apply_reservations(self, reservations: dict[str, int]):
        """
        根据 Space Scheduler 的分配结果，标记 reserved blocks。

        reserved blocks 只能被对应 agent_type 的请求使用。
        shared blocks 可被任何请求使用。
        """
        # 先回收所有 reservation
        all_free = self._get_all_free_blocks()
        self.reserved_blocks.clear()

        # 按分配结果标记
        allocated = 0
        for agent_type, num_blocks in reservations.items():
            reserved = set()
            for block_id in all_free:
                if len(reserved) >= num_blocks:
                    break
                reserved.add(block_id)
            self.reserved_blocks[agent_type] = reserved
            allocated += len(reserved)
            all_free -= reserved

        self.shared_blocks = all_free

    def allocate_slots_with_reservation(
        self,
        request: "Request",
        num_new_tokens: int,
        **kwargs,
    ) -> "KVCacheBlocks | None":
        """
        优先从该 request 的 agent_type reserved pool 分配，
        不足时从 shared pool 补充。
        """
        agent_type = request.agent_type

        # 1. 尝试从 reserved pool 分配
        if agent_type and agent_type in self.reserved_blocks:
            result = self._allocate_from_pool(
                self.reserved_blocks[agent_type],
                request, num_new_tokens, **kwargs,
            )
            if result is not None:
                return result

        # 2. 从 shared pool 分配（fallback）
        return self.allocate_slots(request, num_new_tokens, **kwargs)
```

### 3.2 抢占时的分区保护

```python
# scheduler.py — 抢占逻辑扩展

def _select_preempt_victim(self) -> "Request":
    """
    选择抢占目标时，优先抢占非关键 agent 的请求。

    顺序:
    1. STALLED + NONE（KV 在 HBM 但不推理的非关键请求）
    2. RUNNING 的非关键请求
    3. RUNNING 的关键请求（最后手段）
    """
    critical = self.space_scheduler.critical_agents

    # 候选分组
    stalled_non_critical = []
    running_non_critical = []
    running_critical = []

    for req in self.running:
        if req.status == RequestStatus.STALLED_ON_FC and \
           req.offload_status == OffloadStatus.NONE:
            if req.agent_type not in critical:
                stalled_non_critical.append(req)
        elif req.status == RequestStatus.RUNNING:
            if req.agent_type not in critical:
                running_non_critical.append(req)
            else:
                running_critical.append(req)

    # 按优先级（高 priority 值 = 低优先级 = 优先被抢占）
    for group in [stalled_non_critical, running_non_critical, running_critical]:
        if group:
            return max(group, key=lambda r: (r.priority, r.arrival_time))

    # 不应到达此处
    return max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

---

## 4. 配置

```python
@dataclass
class SpaceSchedulerConfig:
    """Space Scheduler 配置。"""

    # 开关
    priority_enabled: bool = True

    # 优先级参数
    w_static: float = 10.0            # 静态优先级权重

    # 内存分区参数
    initial_reserve_ratio: float = 0.1   # 初始 reserved pool 占比
    min_reserve_ratio: float = 0.0       # reserved pool 最小占比
    max_reserve_ratio: float = 0.4       # reserved pool 最大占比
    gpu_usage_high: float = 0.85         # 高水位阈值
    gpu_usage_low: float = 0.60          # 低水位阈值
    adjustment_step: float = 0.05        # 每次调整步长

    # 关键 Agent 选择
    critical_ratio: float = 0.5          # top 50% agent_type 为关键

    # 更新频率
    update_interval_steps: int = 10      # 每 10 个调度循环更新一次分区
```

---

## 5. 与 PriorityRequestQueue 的集成

不需要修改 `PriorityRequestQueue` 的实现。
Space Scheduler 通过修改 `Request.priority` 字段来影响排序：

```python
# scheduler.py — add_request() 扩展

def add_request(self, request: Request):
    # TokenCake: 注入混合优先级
    if self.tokencake_enabled:
        self.space_scheduler.update_request_priority(request)

    # 现有逻辑
    self.waiting.add_request(request)
```

由于 `Request.__lt__` 已经按 `(priority, arrival_time, request_id)` 排序，
修改 `priority` 字段即可自动影响 `PriorityRequestQueue` 的堆排序。

---

## 6. 调度循环中的周期性更新

```python
# scheduler.py — schedule() 中

def schedule(self) -> SchedulerOutput:
    # ... TokenCake agent events ...

    # 周期性更新内存分区
    self._tokencake_step_counter += 1
    if (self.tokencake_enabled and
        self._tokencake_step_counter % self.space_scheduler.config.update_interval_steps == 0):
        self._update_space_scheduler()

    # ... 现有调度逻辑 ...


def _update_space_scheduler(self):
    """周期性更新 Space Scheduler 状态。"""
    # 1. 更新各 agent_type 的统计
    agent_block_counts = defaultdict(int)
    agent_req_counts = defaultdict(int)
    for req in self.running:
        if req.agent_type:
            blocks = self._get_request_num_blocks(req)
            agent_block_counts[req.agent_type] += blocks
            agent_req_counts[req.agent_type] += 1

    for agent_type in agent_block_counts:
        self.space_scheduler.update_agent_stats(
            agent_type,
            agent_block_counts[agent_type],
            agent_req_counts[agent_type],
        )

    # 2. 选择关键 agent
    self.space_scheduler.select_critical_agents()

    # 3. 更新内存分区
    total = self.kv_cache_manager.get_total_blocks()
    used = self.kv_cache_manager.get_used_blocks()
    reservations = self.space_scheduler.update_memory_reservations(total, used)

    # 4. 应用分区到 KVCacheManager
    if reservations:
        self.kv_cache_manager.apply_reservations(reservations)
```

---

## 7. 改动文件清单

| 文件 | 改动 |
|------|------|
| `vllm/v1/core/sched/space_scheduler.py` | **新建** — SpaceScheduler 类、配置 |
| `vllm/v1/core/sched/scheduler.py` | 引入 SpaceScheduler，周期性更新，抢占保护 |
| `vllm/v1/core/kv_cache_manager.py` | 新增 apply_reservations()、reserve 分区逻辑 |
| `vllm/v1/request.py` | 已在 02 中添加 agent_type、static_priority 字段 |
| `vllm/config.py` | 新增 SpaceSchedulerConfig |
