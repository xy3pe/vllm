# 02 — Request 状态机扩展

> 在现有 RequestStatus 中新增 `STALLED_ON_FC` 状态，
> 使 Scheduler 能识别"暂停等待函数调用"的请求并做差异化处理。

---

## 1. 现有状态机

```
当前 vLLM v1 RequestStatus:

  WAITING ──────────▶ RUNNING ──────────▶ FINISHED_*
     ▲                   │
     │                   │ (KV 不足)
     │                   ▼
     └──────────── PREEMPTED

  WAITING_FOR_FSM ──▶ WAITING  (结构化输出 FSM 初始化完成)
  WAITING_FOR_REMOTE_KVS ──▶ WAITING  (远端 KV 加载完成)
```

---

## 2. 扩展后的状态机

```
  WAITING ──────────▶ RUNNING ──────────▶ FINISHED_*
     ▲                   │ ▲
     │                   │ │ call_finish
     │    (KV 不足)       │ │ + upload 完成
     │                   ▼ │
     │               STALLED_ON_FC ─────┐
     │                   │              │
     │                   │ (KV 不足     │ offload 完成后
     │                   │  且被抢占)    │ blocks 释放给
     │                   ▼              │ 其他请求使用
     └──────────── PREEMPTED            │
                                        ▼
                                   [其他请求得到
                                    释放的 HBM blocks]
```

---

## 3. RequestStatus 枚举扩展

```python
# vllm/v1/request.py — RequestStatus 扩展

class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = 0
    WAITING_FOR_FSM = 1
    WAITING_FOR_REMOTE_KVS = 2
    RUNNING = 3
    STALLED_ON_FC = 4              # ← 新增：等待函数调用完成
    PREEMPTED = 5
    # --- finished statuses ---
    FINISHED_STOPPED = 6
    FINISHED_LENGTH_CAPPED = 7
    FINISHED_ABORTED = 8
    FINISHED_IGNORED = 9
    FINISHED_ERROR = 10

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def is_active(status: "RequestStatus") -> bool:
        """请求仍在系统中（未结束）。"""
        return not RequestStatus.is_finished(status)

    @staticmethod
    def is_schedulable(status: "RequestStatus") -> bool:
        """请求可参与调度（RUNNING 或 STALLED 但 KV 在 HBM 中）。"""
        return status in (RequestStatus.RUNNING, RequestStatus.STALLED_ON_FC)
```

---

## 4. Request 新增字段

```python
# vllm/v1/request.py — Request 类新增字段

@dataclass
class Request:
    # ... 现有字段 ...

    # ---- TokenCake 扩展字段 ----
    agent_type: str | None = None               # Agent 类型标识
    static_priority: float = 0.0                # DAG 关键路径优先级（上层传入）

    # 函数调用状态
    fc_stall_start_time: float | None = None    # call_start 时间戳
    fc_type: str | None = None                  # 当前函数调用类型
    fc_predict_time: float | None = None        # 预测的函数调用时长
    fc_num_stalls: int = 0                      # 累计函数调用次数

    # Offload 状态
    offload_status: OffloadStatus = OffloadStatus.NONE
    offload_start_time: float | None = None     # offload 开始时间
    upload_start_time: float | None = None      # upload 开始时间
```

---

## 5. Offload 状态枚举

```python
# vllm/v1/request.py 新增

class OffloadStatus(enum.IntEnum):
    """KV Cache offload 状态。"""
    NONE = 0                # 未 offload，KV 在 HBM
    OFFLOADING = 1          # 正在 HBM → Host 传输中
    OFFLOADED = 2           # KV 已在 Host RAM（或 UCM Store）
    UPLOADING = 3           # 正在 Host → HBM 传输中（predictive upload）
    UPLOAD_COMPLETE = 4     # upload 完成，等待恢复调度
```

状态转换：

```
NONE ──(should_offload=True)──▶ OFFLOADING ──(传输完成)──▶ OFFLOADED
                                                              │
                                          (predictive upload 触发) │
                                                              ▼
NONE ◀──(call_finish + upload 完成)── UPLOAD_COMPLETE ◀── UPLOADING
```

---

## 6. Scheduler 对 STALLED_ON_FC 的处理

### 6.1 schedule() 主循环改动

```python
# scheduler.py — schedule() 方法

def schedule(self) -> SchedulerOutput:
    # ---- Phase 0: 处理 Agent 事件 ----
    self._process_agent_events()

    # ---- Phase 0.5: 处理 predictive upload ----
    self._check_predictive_uploads()

    # ---- Phase 1: Schedule Running (现有逻辑) ----
    for req in self.running:
        if req.status == RequestStatus.STALLED_ON_FC:
            # STALLED 请求不分配新 tokens，但保留在 running 列表中
            # （除非其 KV 已被 offload，此时不占 running 名额）
            if req.offload_status in (OffloadStatus.OFFLOADED,
                                       OffloadStatus.OFFLOADING):
                # KV 不在 HBM，不算 running 名额
                continue
            else:
                # KV 仍在 HBM（offload 决策为 No），占 running 名额但不调度 tokens
                scheduled_running_reqs.append(req)  # 保持在 batch 中
                continue

        # ... 正常调度逻辑 ...

    # ---- Phase 2: Schedule Waiting (现有逻辑) ----
    # 利用 STALLED+OFFLOADED 腾出的名额和 blocks 调度新请求
    ...
```

### 6.2 _handle_call_start()

```python
def _handle_call_start(self, event: AgentCallStartEvent):
    req = self.requests.get(event.request_id)
    if req is None or req.status != RequestStatus.RUNNING:
        return  # 请求不存在或不在运行中

    # 更新状态
    req.status = RequestStatus.STALLED_ON_FC
    req.fc_stall_start_time = event.timestamp
    req.fc_type = event.fc_type
    req.fc_predict_time = event.predict_time
    req.fc_num_stalls += 1

    # Time Scheduler 决策
    if self.time_scheduler.should_offload(req):
        req.offload_status = OffloadStatus.OFFLOADING
        self._initiate_offload(req)
```

### 6.3 _handle_call_finish()

```python
def _handle_call_finish(self, event: AgentCallFinishEvent):
    req = self.requests.get(event.request_id)
    if req is None or req.status != RequestStatus.STALLED_ON_FC:
        return

    # 更新 EWMA 预测模型
    self.time_scheduler.update_prediction(
        fc_type=req.fc_type,
        actual_duration=event.actual_duration,
    )

    # 根据 offload 状态决定恢复方式
    if req.offload_status == OffloadStatus.NONE:
        # KV 从未离开 HBM，直接恢复
        req.status = RequestStatus.RUNNING
    elif req.offload_status == OffloadStatus.UPLOAD_COMPLETE:
        # predictive upload 已完成，直接恢复
        req.status = RequestStatus.RUNNING
        req.offload_status = OffloadStatus.NONE
    elif req.offload_status == OffloadStatus.UPLOADING:
        # upload 进行中，等 upload 完成后自动恢复
        # （由 _check_predictive_uploads() 处理）
        pass
    elif req.offload_status == OffloadStatus.OFFLOADED:
        # KV 在 Host，需要紧急 upload
        req.offload_status = OffloadStatus.UPLOADING
        self._initiate_upload(req)
    elif req.offload_status == OffloadStatus.OFFLOADING:
        # offload 进行中但函数调用已结束（预测偏长）
        # 取消 offload 或等待完成后立即 upload
        self._cancel_or_reverse_offload(req)

    # 清理函数调用状态
    req.fc_stall_start_time = None
    req.fc_type = None
    req.fc_predict_time = None
```

---

## 7. STALLED 请求的资源核算

| 状态组合 | HBM blocks | running 名额 | 可调度 tokens |
|---------|-----------|-------------|-------------|
| STALLED + NONE | 占用 | 占用 | 0 |
| STALLED + OFFLOADING | 占用(传输中) | 释放 | 0 |
| STALLED + OFFLOADED | 释放 | 释放 | 0 |
| STALLED + UPLOADING | 逐步占用 | 释放 | 0 |
| STALLED + UPLOAD_COMPLETE | 占用 | 占用 | 0 (等待恢复) |
| RUNNING (恢复后) | 占用 | 占用 | 正常 |

**关键收益**：STALLED + OFFLOADED 状态下，该请求的 HBM blocks 和 running 名额都释放，
Scheduler 可以调度更多新请求。

---

## 8. 与现有抢占机制的交互

STALLED 请求也可能被抢占（当 HBM 极度紧张时）：

```python
# 抢占优先级：STALLED_ON_FC 请求优先被抢占（因为它们反正没在推理）
def _select_preempt_victim(self) -> Request:
    # 优先从 STALLED + NONE (KV在HBM但不推理) 中选择
    stalled_in_hbm = [r for r in self.running
                      if r.status == RequestStatus.STALLED_ON_FC
                      and r.offload_status == OffloadStatus.NONE]
    if stalled_in_hbm:
        return max(stalled_in_hbm, key=lambda r: (r.priority, r.arrival_time))

    # 然后从正常 RUNNING 中按现有逻辑选择
    return max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

这与 TokenCake 的 offload 机制互补：
- **offload** = 主动卸载，平滑地释放 HBM
- **preempt** = 被动驱逐，紧急释放 HBM（最后手段）

---

## 9. 改动文件清单

| 文件 | 改动 |
|------|------|
| `vllm/v1/request.py` | 新增 STALLED_ON_FC 状态、OffloadStatus 枚举、Agent 相关字段 |
| `vllm/v1/core/sched/scheduler.py` | schedule() 处理 STALLED 请求、事件处理方法、抢占优先级 |
| `vllm/v1/core/sched/interface.py` | 新增 call_start/call_finish 抽象方法签名 |
| `vllm/v1/engine/core.py` | 新增事件类型，event queue 处理 |
