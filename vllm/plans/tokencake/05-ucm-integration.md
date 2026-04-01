# 05 — UCM 协同设计

> TokenCake offload/upload 与 UCMConnector 的集成方案。
> 核心：复用 UCM 的 dump_data/load_data 通路实现 KV Cache 的暂存与恢复。

---

## 1. 问题定义

TokenCake 需要在函数调用期间将 KV Cache 从 HBM 移走，有三种目标存储：

| 存储层级 | 延迟 | 容量 | 持久性 | 跨实例复用 |
|---------|------|------|--------|-----------|
| Host RAM (CPU) | ~ms | 大 | 否 | 否 |
| UCM Local Store (NFS) | ~10ms | 很大 | 是 | 同节点 |
| UCM Remote Store (Mooncake) | ~100ms | 巨大 | 是 | 跨节点 |

**选择策略**：根据 `predict_time` 选择存储层级。

```
predict_time < 5s   → Host RAM（快进快出）
predict_time 5-60s  → UCM Local Store（可持久化复用）
predict_time > 60s  → UCM Remote Store（长期保存）
```

---

## 2. 架构：两种 Offload 路径统一

```
                    TimeScheduler
                    should_offload() = True
                         │
                         ▼
                  ┌──────────────┐
                  │OffloadRouter │  根据 predict_time 和配置选路
                  └──────┬───────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌────────────┐
        │HostRAM   │ │UCM     │ │UCM Remote  │
        │Offload   │ │Local   │ │Store       │
        │(原生     │ │Store   │ │(Mooncake)  │
        │OffloadMgr)│ │(NFS)   │ │            │
        └────┬─────┘ └───┬────┘ └─────┬──────┘
             │           │            │
             ▼           ▼            ▼
        ┌──────────────────────────────────────┐
        │ Worker 侧: 统一 TransferSpec 执行      │
        │ OffloadingWorker / UCMConnector Worker │
        └──────────────────────────────────────┘
```

---

## 3. OffloadRouter 设计

```python
# vllm/v1/core/sched/offload_router.py (新建)

from enum import IntEnum


class OffloadTarget(IntEnum):
    HOST_RAM = 0       # 原生 OffloadingManager 路径
    UCM_LOCAL = 1      # UCM Store (NFS)
    UCM_REMOTE = 2     # UCM Store (Mooncake)


class OffloadRouter:
    """
    根据函数调用预测时长和系统状态，选择 offload 目标存储。
    """

    def __init__(self, config: "OffloadRouterConfig"):
        self.config = config
        self.has_ucm = False          # 是否配置了 UCMConnector
        self.has_host_offload = False  # 是否配置了 OffloadingManager

    def initialize(
        self,
        connector: "KVConnectorBase_V1 | None",
        offloading_manager: "OffloadingManager | None",
    ):
        """根据部署配置确定可用路径。"""
        self.has_ucm = (connector is not None and
                        hasattr(connector, 'prepare_offload'))
        self.has_host_offload = offloading_manager is not None

    def select_target(
        self,
        request: "Request",
        predict_time: float,
    ) -> OffloadTarget | None:
        """
        选择 offload 目标。

        优先级:
        1. predict_time < host_threshold → Host RAM（最快）
        2. predict_time < ucm_local_threshold → UCM Local
        3. predict_time >= ucm_local_threshold → UCM Remote
        4. 都不可用 → None（不 offload）
        """
        if predict_time < self.config.host_ram_threshold:
            if self.has_host_offload:
                return OffloadTarget.HOST_RAM
            elif self.has_ucm:
                return OffloadTarget.UCM_LOCAL
        elif predict_time < self.config.ucm_remote_threshold:
            if self.has_ucm:
                return OffloadTarget.UCM_LOCAL
            elif self.has_host_offload:
                return OffloadTarget.HOST_RAM
        else:
            if self.has_ucm:
                return OffloadTarget.UCM_REMOTE
            elif self.has_host_offload:
                return OffloadTarget.HOST_RAM

        return None  # 无可用路径


@dataclass
class OffloadRouterConfig:
    host_ram_threshold: float = 5.0       # < 5s → Host RAM
    ucm_remote_threshold: float = 60.0    # > 60s → UCM Remote
```

---

## 4. UCMConnector Scheduler 侧扩展

### 4.1 新增 prepare_offload / prepare_upload 方法

```python
# ucm_connector.py (vllm-ascend) — UCMConnectorV1 扩展

class UCMConnectorV1(KVConnectorBase_V1):
    # ... 现有方法 ...

    def prepare_offload(
        self,
        request: "Request",
        block_hashes: list,
        gpu_block_ids: list[int],
    ) -> "OffloadSpec":
        """
        准备将请求的 KV blocks 卸载到 UCM Store。

        被 TimeScheduler 触发，在 build_connector_meta() 时打包。
        不直接执行传输，而是记录 offload 意图。
        """
        ucm_block_ids = self._hash_blocks(block_hashes)

        spec = OffloadSpec(
            request_id=request.request_id,
            ucm_block_ids=ucm_block_ids,
            gpu_block_ids=gpu_block_ids,
            target=OffloadTarget.UCM_LOCAL,
        )

        # 记录到待执行队列
        self._pending_offloads[request.request_id] = spec
        return spec

    def prepare_upload(
        self,
        request: "Request",
        block_hashes: list,
        target_gpu_block_ids: list[int],
    ) -> "UploadSpec":
        """
        准备将请求的 KV blocks 从 UCM Store 上传回 HBM。

        被 predictive_upload 触发。
        """
        ucm_block_ids = self._hash_blocks(block_hashes)

        spec = UploadSpec(
            request_id=request.request_id,
            ucm_block_ids=ucm_block_ids,
            gpu_block_ids=target_gpu_block_ids,
        )

        self._pending_uploads[request.request_id] = spec
        return spec
```

### 4.2 扩展 build_connector_meta()

```python
def build_connector_meta(
    self,
    scheduler_output: "SchedulerOutput",
) -> "KVConnectorMetadata":
    """
    在现有 load/dump 之外，加入 offload/upload specs。
    """
    # ---- 现有逻辑：处理 scheduled_new_reqs, scheduled_cached_reqs ----
    metadata = self._build_existing_meta(scheduler_output)

    # ---- TokenCake 扩展：加入 offload/upload 指令 ----
    if self._pending_offloads:
        metadata.offload_specs = dict(self._pending_offloads)
        self._pending_offloads.clear()

    if self._pending_uploads:
        metadata.upload_specs = dict(self._pending_uploads)
        self._pending_uploads.clear()

    return metadata
```

### 4.3 扩展 ConnectorMetadata

```python
@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)

    # TokenCake 扩展
    offload_specs: dict[str, OffloadSpec] = field(default_factory=dict)
    upload_specs: dict[str, UploadSpec] = field(default_factory=dict)


@dataclass
class OffloadSpec:
    """HBM → UCM Store 卸载指令。"""
    request_id: str
    ucm_block_ids: list[bytes]
    gpu_block_ids: list[int]
    target: OffloadTarget = OffloadTarget.UCM_LOCAL


@dataclass
class UploadSpec:
    """UCM Store → HBM 上传指令。"""
    request_id: str
    ucm_block_ids: list[bytes]
    gpu_block_ids: list[int]
```

---

## 5. UCMConnector Worker 侧扩展

### 5.1 处理 offload/upload specs

```python
# ucm_connector.py Worker 侧

class UCMDirectConnector:
    # ... 现有方法 ...

    def execute_offload(self) -> None:
        """
        执行 HBM → UCM Store 卸载。

        在 wait_for_save() 之后调用，或作为独立步骤。
        """
        metadata = self._get_connector_metadata()
        if not hasattr(metadata, 'offload_specs') or not metadata.offload_specs:
            return

        for req_id, spec in metadata.offload_specs.items():
            # 提取 GPU 地址
            total_ptrs = self.kv_cache_layout.extract_block_addrs(
                spec.gpu_block_ids
            )

            # 获取同步 event handle
            event_handle = self._get_dump_event_handle()

            # 异步 dump 到 UCM Store
            task = self.store.dump_data(
                spec.ucm_block_ids,
                self.shard_indexs,
                total_ptrs,
                event_handle,
            )

            # 非阻塞：offload 可以后台完成
            self._offload_tasks[req_id] = task

    def execute_upload(self) -> None:
        """
        执行 UCM Store → HBM 上传。

        在 start_load_kv() 之前调用。
        """
        metadata = self._get_connector_metadata()
        if not hasattr(metadata, 'upload_specs') or not metadata.upload_specs:
            return

        for req_id, spec in metadata.upload_specs.items():
            total_ptrs = self.kv_cache_layout.extract_block_addrs(
                spec.gpu_block_ids
            )

            task = self.store.load_data(
                spec.ucm_block_ids,
                self.shard_indexs,
                total_ptrs,
            )

            # Upload 需要在 forward 前完成
            self.store.wait(task)

    def check_offload_completion(self) -> dict[str, bool]:
        """
        检查后台 offload 任务完成状态。
        返回 {request_id: completed}。
        """
        completed = {}
        for req_id, task in list(self._offload_tasks.items()):
            if self.store.check(task):
                completed[req_id] = True
                del self._offload_tasks[req_id]
            else:
                completed[req_id] = False
        return completed
```

---

## 6. 与 OffloadingConnector 的协同

当同时配置了 vLLM 原生 OffloadingManager 和 UCMConnector 时：

```
                  OffloadRouter
                       │
            ┌──────────┼──────────┐
            ▼                     ▼
    OffloadingConnector      UCMConnector
    (Host RAM 快速通路)      (持久化通路)
            │                     │
            ▼                     ▼
       CPU Block Pool       UCM Store (NFS/Remote)
```

### 6.1 Host RAM 路径（短暂 offload）

```python
# scheduler.py

def _initiate_offload_host(self, request: "Request"):
    """通过 OffloadingManager 卸载到 Host RAM。"""
    block_hashes = self._get_block_hashes(request)
    store_output = self.offloading_manager.prepare_store(block_hashes)
    if store_output is None:
        return False  # Host RAM 不足

    # 加入 connector metadata
    self._pending_host_offloads[request.request_id] = store_output
    request.offload_status = OffloadStatus.OFFLOADING
    return True
```

### 6.2 UCM 路径（持久化 offload）

```python
def _initiate_offload_ucm(self, request: "Request"):
    """通过 UCMConnector 卸载到 UCM Store。"""
    block_hashes = self._get_block_hashes(request)
    gpu_block_ids = self._get_gpu_block_ids(request)

    spec = self.connector.prepare_offload(
        request, block_hashes, gpu_block_ids
    )
    request.offload_status = OffloadStatus.OFFLOADING
    return True
```

### 6.3 统一 _initiate_offload()

```python
def _initiate_offload(self, request: "Request"):
    """统一 offload 入口。"""
    predict_time = self.time_scheduler.predict_fc_duration(
        request.fc_type, request.fc_predict_time
    )
    target = self.offload_router.select_target(request, predict_time)

    if target is None:
        return False

    if target == OffloadTarget.HOST_RAM:
        return self._initiate_offload_host(request)
    else:
        return self._initiate_offload_ucm(request)
```

---

## 7. UCM 前缀复用优化

当 offloaded KV 已在 UCM Store 中时，恢复可以更快：

```python
# scheduler.py — _initiate_upload() 中

def _initiate_upload(self, request: "Request"):
    """恢复 KV 到 HBM。"""

    if request.offload_target == OffloadTarget.HOST_RAM:
        # Host RAM → HBM：通过 OffloadingManager
        block_hashes = self._get_block_hashes(request)
        self.offloading_manager.prepare_load(block_hashes)

    elif request.offload_target in (OffloadTarget.UCM_LOCAL,
                                     OffloadTarget.UCM_REMOTE):
        # UCM Store → HBM：通过 UCMConnector
        # 优化：检查 UCM 是否已有该 KV（prefix cache hit）
        block_hashes = self._get_block_hashes(request)
        gpu_block_ids = self.kv_cache_manager.allocate_blocks_for_upload(
            request, len(block_hashes)
        )

        if gpu_block_ids is None:
            # HBM blocks 不够（gradual reservation 还没完成）
            return False

        self.connector.prepare_upload(request, block_hashes, gpu_block_ids)

    request.offload_status = OffloadStatus.UPLOADING
    return True
```

---

## 8. 跨实例 KV 复用场景

当多个 vLLM 实例服务同一个多 Agent 应用时，
offloaded KV 存入 UCM Remote Store 后，其他实例可以直接读取：

```
Instance A:  Agent-Programmer → tool_call → offload KV to UCM Remote
                                                    │
                                                    ▼ (Mooncake)
Instance B:  Agent-Programmer (相同 prompt) → prefix cache HIT → skip prefill
```

这通过 UCMConnector 现有的 `get_num_new_matched_tokens()` 自动实现，
无需额外代码。TokenCake 的 offload 行为天然增加了 UCM Store 中的 KV 缓存，
提升了跨实例前缀命中率。

---

## 9. NpuDevice 事件同步

Ascend NPU 上的异步传输需要 ACL 事件同步：

```python
# 复用 UCMConnector 现有的 NpuDevice 机制

class NpuDevice:
    def get_event_handle(self) -> int:
        """
        在 NPU compute stream 上记录事件，
        返回 event handle 给 UCM Store 的 C++ 层。
        C++ storage stream 等待该事件后再开始 D2H 传输。
        """
        event = acl.rt.create_event()
        acl.rt.record_event(event, self.compute_stream)
        return event

    def synchronize(self):
        torch.npu.current_stream().synchronize()
```

TokenCake 的 offload/upload 传输直接复用这套机制，无需新增同步原语。

---

## 10. 改动文件清单

| 文件 | 改动 |
|------|------|
| `vllm/v1/core/sched/offload_router.py` | **新建** — OffloadRouter、OffloadTarget |
| `vllm/v1/core/sched/scheduler.py` | _initiate_offload/upload 统一入口 |
| `vllm_ascend/distributed/ucm_connector.py` | prepare_offload/upload、扩展 build_connector_meta |
| `ucm/integration/vllm/ucm_connector.py` | execute_offload/upload Worker 侧实现 |
| `vllm/v1/core/sched/output.py` | SchedulerOutput.kv_connector_metadata 携带 offload/upload specs |
