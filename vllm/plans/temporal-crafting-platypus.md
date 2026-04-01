# Plan: 将被动卸载（Eviction Offload）合并入 OffloadingConnector 框架

## Context

当前被动卸载（auto-swap）使用独立的 `release_offloading_manager` + `collective_rpc("offload_release_blocks")` 路径，与 vLLM 已有的 `OffloadingConnector` 框架并行存在。两者功能重叠：都是将 KV cache block 从 GPU/NPU 卸载到 CPU，命中时恢复。

**目标**：将被动卸载合并入 OffloadingConnector，通过 `offload_strategy` 参数区分：
- `"proactive"`（现有默认）：每计算一步就往 CPU 写一份
- `"eviction"`（新增）：仅在 block 被淘汰时抢救到 CPU

合并后，两种策略共享 OffloadingManager、CPUBackend、Worker 传输 Handler 和 restore 路径。

## 核心挑战：时序约束

```
proactive 策略：
  execute_model() → 数据已计算 → wait_for_save() → start_store_kv() → 异步 GPU→CPU
  ✓ 数据始终有效，异步传输安全

eviction 策略：
  schedule() → block 被淘汰 → block ID 被重新分配给新请求
  → execute_model() → 覆盖 block 数据 ← 必须在此之前完成 GPU→CPU
  ✗ 不能用 wait_for_save()（在 execute_model 之后）
```

**解法**：eviction store 放在 `start_load_kv()`（execute_model 之前）处理，而非 `wait_for_save()`（之后）。

## 实现步骤

### Step 1: 添加 `offload_strategy` 配置

**文件**: `vllm/v1/kv_offload/cpu.py` (CPUOffloadingSpec)

```python
self.offload_strategy: str = self.extra_config.get("offload_strategy", "proactive")
# eviction 策略强制 block_size_factor = 1（单块淘汰，不分组）
if self.offload_strategy == "eviction":
    self.offloaded_block_size = self.gpu_block_size
```

使用方式（启动参数）：
```
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "num_cpu_blocks": 5000,
    "offload_strategy": "eviction"
  }
}'
```

### Step 2: 扩展 OffloadingConnectorMetadata

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

```python
@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    eviction_stores: list[TransferSpec]  # 新增：eviction 策略的 store
```

### Step 3: 添加 `notify_evictions()` 到 OffloadingConnectorScheduler

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

```python
class OffloadingConnectorScheduler:
    def __init__(self, spec: OffloadingSpec):
        ...
        self.offload_strategy = getattr(spec, 'offload_strategy', 'proactive')
        self._pending_eviction_stores: list[TransferSpec] = []

    def notify_evictions(self, evictions: list[tuple[int, BlockHash]]):
        """Called by scheduler after schedule() with evicted block info."""
        if not evictions or self.offload_strategy != "eviction":
            return

        gpu_block_ids = [block_id for block_id, _ in evictions]
        block_hashes = [bh for _, bh in evictions]

        store_output = self.manager.prepare_store(block_hashes)
        if store_output is None or not store_output.block_hashes_to_store:
            return

        # Map stored hashes back to GPU block IDs
        hash_to_gpu_id = dict(zip(block_hashes, gpu_block_ids))
        src_block_ids = [hash_to_gpu_id[h] for h in store_output.block_hashes_to_store]

        src_spec = GPULoadStoreSpec(src_block_ids)
        dst_spec = store_output.store_spec
        self._pending_eviction_stores.append((src_spec, dst_spec))

        # Eviction stores are synchronous, mark complete immediately
        self.manager.complete_store(store_output.block_hashes_to_store)
```

### Step 4: 修改 `_get_reqs_to_store()` —— eviction 策略不做 proactive store

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

在 `_get_reqs_to_store()` 开头加判断：
```python
def _get_reqs_to_store(self, scheduler_output):
    if self.offload_strategy == "eviction":
        return {}  # eviction 策略不做 proactive store
    # ... 原有 proactive 逻辑不变
```

### Step 5: 修改 `build_connector_meta()` 包含 eviction stores

```python
def build_connector_meta(self, scheduler_output):
    meta = OffloadingConnectorMetadata(
        reqs_to_load=self._reqs_to_load,
        reqs_to_store=self._get_reqs_to_store(scheduler_output),
        eviction_stores=self._pending_eviction_stores,  # 新增
    )
    self._reqs_to_load = {}
    self._pending_eviction_stores = []
    return meta
```

### Step 6: Worker 侧 —— 在 `start_load_kv()` 中处理 eviction stores

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

修改 `OffloadingConnectorWorker.start_load_kv()`：
```python
def start_load_kv(self, metadata: OffloadingConnectorMetadata):
    # 先处理 eviction stores（必须在 execute_model 之前完成）
    for transfer_spec in metadata.eviction_stores:
        job_id = self._generate_job_id()
        # 同步执行：block 即将被覆盖，不能异步
        self.worker.transfer_sync(job_id, transfer_spec)

    # 再处理正常的 load
    for req_id, transfer_spec in metadata.reqs_to_load.items():
        ...
```

> **注意**：`OffloadingWorker` 当前只有 `transfer_async()`。需要增加 `transfer_sync()` 或直接在 eviction store 调用 async 后立即 wait。

### Step 7: 添加 `notify_evictions()` 到 KVConnectorBase_V1

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

```python
def notify_evictions(self, evictions: list[tuple[int, "BlockHash"]]):
    """Notify connector of blocks evicted from prefix cache.
    Default: no-op. Override in connectors that support eviction offload."""
    pass
```

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

在 `OffloadingConnector` 中重写：
```python
def notify_evictions(self, evictions):
    if self.connector_scheduler:
        self.connector_scheduler.notify_evictions(evictions)
```

### Step 8: Scheduler 调用 connector.notify_evictions()

**文件**: `vllm/v1/core/sched/scheduler.py`

在 `schedule()` 末尾（现有 `_process_evicted_blocks_for_swap()` 的位置），改为：
```python
if self.connector is not None:
    evictions = self.kv_cache_manager.take_pending_evictions()
    if evictions:
        # 将 BlockHashWithGroupId → BlockHash
        from vllm.v1.core.kv_cache_utils import get_block_hash
        converted = [(bid, get_block_hash(bh)) for bid, bh in evictions]
        self.connector.notify_evictions(converted)
```

这样无需 `release_offloading_manager` 单独存在——connector 内部的 OffloadingManager 统一管理。

### Step 9: Restore 路径

**无需额外修改**。OffloadingConnector 现有的 `get_num_new_matched_tokens()` → `update_state_after_alloc()` → `start_load_kv()` 路径已经处理 CPU→GPU restore，且是异步的。eviction 策略存入的 block 和 proactive 策略存入的 block 在同一个 OffloadingManager 中，restore 路径自动覆盖。

## OffloadingWorker 同步传输

**文件**: `vllm/v1/kv_offload/worker/worker.py`

需要确认 `transfer_async` + 立即等待完成 是否可行。如果 `OffloadingWorker` 使用 CUDA stream，可以：
```python
def start_load_kv(self, metadata):
    # Eviction stores: submit + wait
    for transfer_spec in metadata.eviction_stores:
        job_id = self._generate_job_id()
        self.worker.transfer_async(job_id, transfer_spec)
    # Wait for all eviction stores to complete
    while any pending eviction jobs:
        self.worker.get_finished()  # blocking
    # Then proceed with loads
    ...
```

或者在 `OffloadingWorker` 中添加 `transfer_sync()` 方法直接调用底层 handler 的 copy 操作。

## 修改文件清单

| 文件 | 修改 |
|------|------|
| `vllm/v1/kv_offload/cpu.py` | 读取 `offload_strategy`，eviction 时强制 `block_size_factor=1` |
| `vllm/v1/kv_offload/spec.py` | 添加 `offload_strategy` 属性（可选） |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 添加 `notify_evictions()` 默认空方法 |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | Metadata 扩展 + `notify_evictions()` + `_get_reqs_to_store` 分支 + worker 处理 eviction stores |
| `vllm/v1/kv_offload/worker/worker.py` | 添加 `transfer_sync()` 或确认 async+wait 可行 |
| `vllm/v1/core/sched/scheduler.py` | `schedule()` 末尾调用 `connector.notify_evictions()` 替代 `_process_evicted_blocks_for_swap()` |

**可移除的代码**（合并后不再需要）：
- `scheduler.py` 中的 `release_offloading_manager`、`_pending_release_transfers`、`_process_evicted_blocks_for_swap()`、`take_pending_release_transfers()`、`lookup_cpu_cache_for_request()`
- `engine/core.py` 中的 `_try_restore_from_cpu_cache()` 和相关 `collective_rpc("offload/load_release_blocks")` 调用
- `gpu_worker.py` / `worker.py (vllm-ascend)` 中的 `_init_release_cpu_caches()`、`offload_release_blocks()`、`load_release_blocks()` —— 被 OffloadingWorker + CpuGpuOffloadingHandlers 替代

## NPU / vllm-ascend 适配

vllm-ascend 已有 `cpu_offload_connector.py` 继承自 OffloadingConnector。合并后：
- `offload_strategy` 配置自动继承
- NPU 侧的 `CpuGpuOffloadingHandlers` 需要适配 NPU tuple kv_caches（现有 vllm-ascend connector 已处理）
- `notify_evictions()` 通过 base class 继承，无需额外适配

## 风险点

1. **时序竞争**：eviction store 必须在 `execute_model()` 之前完成。通过将其放在 `start_load_kv()` 中同步执行来保证。
2. **block_size_factor**：eviction 是单块粒度，proactive 可能是多块分组。eviction 策略强制 `block_size_factor=1`。
3. **release_kv_cache API**：主动卸载（release_kv_cache）保持独立，不受此合并影响。它有独立的 aging 机制和 API 入口。

## 验证方案

1. 启动参数：`--kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"num_cpu_blocks": 5000, "offload_strategy": "eviction"}}'`
2. 发送大量不同请求填满 GPU prefix cache
3. 观察日志：eviction store 发生在 execute_model 之前
4. 发送与早期请求相同前缀的请求
5. 观察日志：CPU cache hit → async restore via OffloadingConnector
6. 同时测试 `offload_strategy: "proactive"` 确认原有功能不受影响
