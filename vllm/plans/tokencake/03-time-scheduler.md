# 03 — Time Scheduler 设计

> 核心组件：决定是否 offload、何时 upload、预测函数调用时长。

---

## 1. 组件定位

```
               call_start 事件
                    │
                    ▼
            ┌──────────────┐
            │TimeScheduler │
            │              │
            │ should_      │──▶ offload 决策 ──▶ _initiate_offload()
            │  offload()   │
            │              │
            │ predict_fc() │──▶ 预测剩余时间
            │              │
            │ check_       │──▶ upload 触发  ──▶ _initiate_upload()
            │  uploads()   │
            │              │
            │ update_      │◀── call_finish 事件
            │  prediction()│
            └──────────────┘
```

TimeScheduler 是一个**纯决策组件**，不直接操作 KV Cache 传输。
它输出决策，由 Scheduler 调用 KVCacheManager / UCMConnector 执行。

---

## 2. 类设计

```python
# vllm/v1/core/sched/time_scheduler.py (新建)

from dataclasses import dataclass, field
from collections import defaultdict
import time
import math


@dataclass
class FCTypeStats:
    """某个 fc_type 的历史统计。"""
    ewma_duration: float = 0.0      # EWMA 均值
    count: int = 0                  # 观测次数
    alpha: float = 1.0              # 开发者估计权重（逐步衰减）
    last_duration: float = 0.0      # 最近一次实际时长
    max_observed: float = 0.0       # 观测到的最大值


# 冷启动默认值
FC_TYPE_COLD_START: dict[str, float] = {
    "web_search":       3.0,
    "local_search":     1.0,
    "user_interaction": 60.0,
    "doc_processing":   5.0,
    "code_execution":   10.0,
    "citation_verify":  30.0,
    "default":          5.0,
}


class TimeScheduler:
    """
    TokenCake Time Scheduler.

    职责：
    1. should_offload() — 决定是否将 STALLED 请求的 KV 卸载到 Host
    2. predict_fc_duration() — 预测函数调用剩余时间
    3. check_predictive_uploads() — 在调度循环中检查是否该触发 upload
    4. update_prediction() — call_finish 时更新 EWMA 模型
    """

    def __init__(self, config: "TimeSchedulerConfig"):
        self.config = config

        # EWMA 模型：fc_type → 统计
        self.fc_stats: dict[str, FCTypeStats] = defaultdict(FCTypeStats)

        # 传输时间模型参数（硬件相关，初始化时标定）
        self.offload_time_per_block: float = config.offload_time_per_block
        self.upload_time_per_block: float = config.upload_time_per_block

    # ================================================================
    # 1. Offload 决策 (Algorithm 1)
    # ================================================================

    def should_offload(self, request: "Request") -> bool:
        """
        决定是否将 STALLED 请求的 KV Cache offload 到 Host/UCM。

        判断逻辑:
            Benefit_scheduling > Overhead_transfer

        即：函数调用的剩余时间扣除传输开销后，
        能否处理至少一个等待请求的 tokens。
        """
        if not self.config.offload_enabled:
            return False

        num_blocks = self._get_request_num_blocks(request)
        T_transfer = self.calculate_transfer_time(num_blocks)
        T_fc = self.predict_fc_duration(
            fc_type=request.fc_type,
            predict_time=request.fc_predict_time,
        )

        # Guard: 函数调用太短，不值得 offload
        if T_fc <= T_transfer * self.config.safety_margin:
            return False

        # 可用调度窗口
        T_window = T_fc - T_transfer

        # 检查等待队列是否有请求能利用该窗口
        # （简化版：只要 T_window > 最小有效窗口即可）
        if T_window < self.config.min_offload_window:
            return False

        return True

    # ================================================================
    # 2. 传输时间估算
    # ================================================================

    def calculate_transfer_time(self, num_blocks: int) -> float:
        """
        估算 offload + upload 的往返传输时间。

        T_transfer = T_offload(N) + T_upload(N)

        线性模型，参数从硬件标定或运行时测量获得。
        """
        T_offload = num_blocks * self.offload_time_per_block
        T_upload = num_blocks * self.upload_time_per_block
        return T_offload + T_upload

    # ================================================================
    # 3. 函数调用时长预测 (Eq. 1)
    # ================================================================

    def predict_fc_duration(
        self,
        fc_type: str,
        predict_time: float | None = None,
    ) -> float:
        """
        预测函数调用总时长。

        三层模型:
          L0: 冷启动默认值（按 fc_type 基类）
          L1: 开发者标注 predict_time
          L2: EWMA 历史均值

        融合公式:
          t_final = alpha * t_dev + (1 - alpha) * t_hist

        alpha 从 1.0 衰减到 alpha_min，随调用次数增加。
        """
        stats = self.fc_stats.get(fc_type)
        fc_base = self._get_fc_base_type(fc_type)

        if stats is None or stats.count == 0:
            # 冷启动：使用开发者估计或默认值
            if predict_time is not None:
                return predict_time
            return FC_TYPE_COLD_START.get(fc_base,
                                          FC_TYPE_COLD_START["default"])

        t_hist = stats.ewma_duration

        if predict_time is not None:
            # 融合开发者估计与历史
            alpha = stats.alpha
            return alpha * predict_time + (1.0 - alpha) * t_hist
        else:
            return t_hist

    # ================================================================
    # 4. Predictive Upload 检查
    # ================================================================

    def check_predictive_uploads(
        self,
        stalled_requests: list["Request"],
    ) -> list["Request"]:
        """
        检查哪些 STALLED+OFFLOADED 请求应该触发 predictive upload。

        在每个调度循环中调用。

        判断逻辑：
          remaining_time = predict_fc_duration - elapsed_time
          T_upload = upload_time_per_block * num_blocks
          if remaining_time <= T_upload + buffer:
              trigger upload

        返回需要触发 upload 的请求列表。
        """
        to_upload = []
        now = time.monotonic()

        for req in stalled_requests:
            if req.offload_status != OffloadStatus.OFFLOADED:
                continue

            elapsed = now - req.fc_stall_start_time
            predicted_total = self.predict_fc_duration(
                fc_type=req.fc_type,
                predict_time=req.fc_predict_time,
            )
            remaining = predicted_total - elapsed

            num_blocks = self._get_request_num_blocks(req)
            T_upload = num_blocks * self.upload_time_per_block
            buffer = self.config.upload_buffer_time

            if remaining <= T_upload + buffer:
                to_upload.append(req)

        return to_upload

    # ================================================================
    # 5. EWMA 更新
    # ================================================================

    def update_prediction(
        self,
        fc_type: str,
        actual_duration: float,
    ):
        """
        call_finish 时更新 EWMA 模型。

        EWMA: t_hist = beta * actual + (1 - beta) * t_hist_old
        alpha 衰减: alpha = max(alpha - decay_rate, alpha_min)
        """
        stats = self.fc_stats[fc_type]

        # 异常值过滤
        if (stats.count > 0 and
            actual_duration > stats.ewma_duration * self.config.outlier_factor):
            # 异常偏长，不更新 EWMA，仅记录
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

        # alpha 衰减
        stats.alpha = max(
            stats.alpha - self.config.alpha_decay_rate,
            self.config.alpha_min,
        )

    # ================================================================
    # 内部辅助
    # ================================================================

    def _get_request_num_blocks(self, request: "Request") -> int:
        """获取请求当前占用的 KV Cache block 数。"""
        # 由 Scheduler 注入或从 KVCacheManager 查询
        return getattr(request, '_num_kv_blocks', 0)

    def _get_fc_base_type(self, fc_type: str) -> str:
        """从 'web_search:tavily' 提取基类 'web_search'。"""
        return fc_type.split(":")[0] if ":" in fc_type else fc_type
```

---

## 3. 配置

```python
# vllm/v1/core/sched/time_scheduler.py

@dataclass
class TimeSchedulerConfig:
    """Time Scheduler 配置参数。"""

    # 开关
    offload_enabled: bool = True

    # EWMA 参数
    ewma_beta: float = 0.3           # 新观测权重
    alpha_decay_rate: float = 0.1    # 每次调用后 alpha 衰减步长
    alpha_min: float = 0.2           # alpha 下限
    outlier_factor: float = 3.0      # 超过 3x 均值视为异常

    # Offload 决策参数
    safety_margin: float = 1.5       # T_fc > T_transfer * safety_margin 才 offload
    min_offload_window: float = 0.5  # 最小有效窗口（秒）

    # Upload 参数
    upload_buffer_time: float = 0.1  # 额外 buffer（秒），提前触发 upload

    # 传输时间模型（硬件相关，需标定）
    # Ascend 910B 参考值（优化后）
    offload_time_per_block: float = 0.001   # 1ms per block (HBM→Host)
    upload_time_per_block: float = 0.001    # 1ms per block (Host→HBM)

    # Gradual GPU Block Reservation
    gradual_reservation_enabled: bool = True
    reservation_cycles: int = 5      # 分多少轮预留 HBM blocks
```

---

## 4. 硬件标定流程

传输时间模型的参数需在初始化时标定：

```python
class TimeScheduler:
    def calibrate_transfer_time(self, block_sizes: list[int] = [64, 256, 1024]):
        """
        启动时标定传输时间模型。

        发送不同大小的 blocks 做 offload/upload，
        线性回归得到 per-block 时间。
        """
        offload_times = []
        upload_times = []
        for n in block_sizes:
            t_off = self._benchmark_offload(n)
            t_up = self._benchmark_upload(n)
            offload_times.append((n, t_off))
            upload_times.append((n, t_up))

        # 线性回归
        self.offload_time_per_block = linear_fit(offload_times)
        self.upload_time_per_block = linear_fit(upload_times)
```

---

## 5. Offload/Upload 执行路径

TimeScheduler 只做决策，实际传输通过两条路径：

### 路径 A：通过现有 OffloadingManager (vLLM 原生)

```
TimeScheduler.should_offload() = True
    │
    ▼
Scheduler._initiate_offload(req)
    │
    ├── offloading_manager.prepare_store(block_hashes)
    │   → 返回 StoreSpec (src_blocks, dst_blocks)
    │
    ├── 将 StoreSpec 加入 connector_metadata
    │
    └── Worker 侧 OffloadingWorker 执行异步传输
```

### 路径 B：通过 UCMConnector (vLLM-Ascend)

```
TimeScheduler.should_offload() = True
    │
    ▼
Scheduler._initiate_offload(req)
    │
    ├── ucm_connector.prepare_offload(request, block_hashes)
    │   → 扩展 UCMConnectorMetadata 加入 offload 指令
    │
    ├── build_connector_meta() 时打包 offload specs
    │
    └── Worker 侧 UCMConnector.execute_offload() 执行
        → store.dump_data() 到 UCM Store
```

具体选择哪条路径取决于部署配置（见 05-ucm-integration.md）。

---

## 6. Predictive Upload 时序图

```
时间 ──────────────────────────────────────────────────────▶

call_start                                    call_finish
   │                                              │
   ▼                                              ▼
   ├── offload 决策 ──▶ HBM→Host 传输 ──▶ OFFLOADED
   │                    (异步，~几ms)
   │
   │   ... 其他请求利用释放的 HBM blocks ...
   │
   │                          predict_finish
   │                              │
   │                              ▼
   │                     ┌─ T_upload + buffer ─┐
   │                     │                     │
   │               trigger upload        upload 完成
   │                     │                     │
   │                     ▼                     ▼
   │               Host→HBM 传输 ────▶ UPLOAD_COMPLETE
   │               (gradual block       │
   │                reservation)        ▼
   │                              恢复 RUNNING
   │
   ├──────── T_fc_predicted ──────────┤
```

理想情况下，upload 在 call_finish 之前刚好完成，实现零等待恢复。

---

## 7. Gradual GPU Block Reservation

避免 upload 时一次性申请大量 HBM blocks 导致调度抖动：

```python
class TimeScheduler:
    def get_gradual_reservation_plan(
        self,
        request: "Request",
        remaining_time: float,
    ) -> list[int]:
        """
        计算每轮调度循环应预留的 HBM block 数。

        将 upload 所需 blocks 均匀分散到 remaining_time 内的多轮循环中。
        """
        total_blocks = self._get_request_num_blocks(request)
        cycles = self.config.reservation_cycles

        blocks_per_cycle = math.ceil(total_blocks / cycles)
        plan = []
        remaining = total_blocks
        for i in range(cycles):
            n = min(blocks_per_cycle, remaining)
            plan.append(n)
            remaining -= n
            if remaining <= 0:
                break
        return plan
```

Scheduler 在每轮 schedule() 中执行一步 reservation：

```python
# scheduler.py
def _step_gradual_reservation(self):
    for req in self._gradual_reservation_queue:
        plan = req._reservation_plan
        if plan:
            n_blocks = plan.pop(0)
            reserved = self.kv_cache_manager.reserve_blocks(req, n_blocks)
            if not reserved:
                # HBM 不够，推迟到下一轮
                plan.insert(0, n_blocks)
```

---

## 8. 改动文件清单

| 文件 | 改动 |
|------|------|
| `vllm/v1/core/sched/time_scheduler.py` | **新建** — TimeScheduler 类、FCTypeStats、配置 |
| `vllm/v1/core/sched/scheduler.py` | 引入 TimeScheduler，调用 should_offload / check_predictive_uploads |
| `vllm/config.py` | 新增 TimeSchedulerConfig 配置项 |
| `vllm/v1/core/kv_cache_manager.py` | 新增 reserve_blocks() 方法（gradual reservation） |
