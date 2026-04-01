# CPU Offload 特性说明

## 概述

CPU Offload 特性在 NPU/GPU 显存之外，利用 CPU pinned memory 作为 KV cache block 的二级缓存。当 block 从显存中被移除时，数据被保存到 CPU 内存；当后续请求的 prefix 命中 CPU 缓存时，数据从 CPU 恢复到显存，跳过重新计算。

该特性包含两个子特性：**被动卸载**和**主动卸载**，共享底层基础设施但触发机制不同。

---

## 子特性一：被动卸载（Auto-Swap）

### 触发条件
NPU 显存不足、prefix cache 中的 block 被淘汰时自动触发。

### 工作流程
```
1. 调度器分配 block 时发现显存不足
2. 从 prefix cache 淘汰旧 block（LRU）
3. 淘汰前：自动将 block 数据拷贝到 CPU（NPU→CPU）
4. 新请求到达，prefix 匹配 CPU 中的 block
5. 自动将 block 数据恢复到 NPU（CPU→NPU）
6. 调度器识别为 cache hit，跳过 prefill 计算
```

### 适用场景
- 高并发压力下显存频繁淘汰 block
- 请求之间存在共享前缀（如相同 system prompt）
- 无需业务层感知，完全由引擎自动管理

### 关键代码路径
| 阶段 | 位置 | 说明 |
|------|------|------|
| 淘汰检测 | `scheduler.py` schedule() | 在 block 被淘汰前收集需要 offload 的 block |
| NPU→CPU | `core.py` step() | 通过 collective_rpc 调用 worker 执行 swap |
| CPU→NPU | `core.py` add_request() | 新请求到达时检查 CPU cache，命中则恢复 |

---

## 子特性二：主动卸载（Release Offload）

### 触发条件
业务层显式调用 `/v1/release_kv_cache` API 时触发。

### 工作流程
```
1. 业务层调用 POST /v1/release_kv_cache（指定 cache_salt + messages）
2. 指定 block 被 aging（标记为优先淘汰）
3. 同时将 block 数据拷贝到 CPU（NPU→CPU）
4. 新请求携带相同 cache_salt 到达
5. 检查 NPU prefix cache，若命中则直接使用（不需要 CPU→NPU）
6. 若 NPU 已淘汰，检查 CPU cache，命中则恢复（CPU→NPU）
```

### 适用场景
- 多轮对话场景：会话间隙释放 KV cache，下轮命中时快速恢复
- PD 分离架构：prefill 节点释放后保留 CPU 备份，减少重复 prefill 计算
- 业务层需要精确控制哪些 block 被保留

### 关键代码路径
| 阶段 | 位置 | 说明 |
|------|------|------|
| API 入口 | `api_server.py` /v1/release_kv_cache | 接收释放请求 |
| Aging + Offload | `scheduler.py` release_kv_cache() | aging block + prepare_store + 收集 transfer spec |
| NPU→CPU | `core.py` release_kv_cache() | 通过 collective_rpc 调用 worker 执行 swap |
| CPU→NPU | `core.py` add_request() | 新请求到达时检查，优先用 NPU cache，miss 时查 CPU |

---

## 共同点

| 维度 | 说明 |
|------|------|
| CPU 内存管理 | 共用 `CPUBackend` + `LRUOffloadingManager`，统一管理 CPU block 分配与淘汰 |
| 数据传输 | 共用 `ops.swap_blocks`（GPU）/ `torch.ops._C_ascend.swap_blocks`（NPU） |
| CPU 内存分配 | 共用 Worker 的 `_init_release_cpu_caches()`，启动时一次性分配 pinned memory |
| 恢复路径 | 共用 `_try_restore_from_cpu_cache()`，新请求到达时统一检查 CPU cache |
| 启用参数 | 共用 `--num-release-cpu-blocks N`，N>0 时两者同时启用 |
| NPU 优先 | 恢复时优先检查 NPU prefix cache，miss 时才查 CPU cache，避免不必要的拷贝 |
| 可观测性 | 共用 cache stats 日志（npu_hit_rate / cpu_hit_rate / block_accesses） |

## 差异点

| 维度 | 被动卸载 | 主动卸载 |
|------|---------|---------|
| 触发方式 | 引擎自动（显存压力淘汰时） | 业务显式调用 API |
| 触发时机 | schedule() 循环中 | release_kv_cache() 请求处理时 |
| 卸载范围 | 被淘汰的 cached block | API 指定的 messages 对应的 block |
| 是否 aging | 不 aging（已经在淘汰流程中） | 先 aging（标记优先淘汰） |
| 业务感知 | 无需业务层感知 | 需要业务层传递 cache_salt |
| 恢复触发 | 任意请求的 prefix 匹配即可 | 任意请求的 prefix 匹配即可 |
| PD 架构价值 | prefill/decode 均有效 | 主要在 prefill 节点有效 |

---

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-release-cpu-blocks` | int | None | CPU block 数量，>0 启用，0 或不设则关闭 |
| `--release-offload-checksum` | bool | False | 启用 offload/load 时的 tensor checksum 校验（调试用） |

## 关键日志

```
# 启动
Release offloading manager initialized with 5000 CPU blocks
Allocated N CPU cache tuples with 5000 blocks each for release KV cache offloading

# 被动卸载（NPU→CPU）
Auto-offloading N evicted blocks to CPU
offload_release_blocks: swapping N blocks NPU->CPU

# 主动卸载（NPU→CPU）
release_kv_cache called: session_id=xxx, num_block_hashes=N, offloading_manager=True
release_kv_cache aged N blocks
offload_release_blocks: swapping N blocks NPU->CPU
Offloaded N transfer batches to CPU

# 恢复（CPU→NPU）
CPU cache hit: N blocks (M tokens) for request xxx
load_release_blocks: swapping N blocks CPU->NPU
Restored N blocks from CPU to GPU for request xxx

# 每请求统计
Request xxx finished. cache stats: {npu_total, npu_allocated, npu_free, npu_cached, cpu_stored, npu_hit_rate, cpu_hit_rate, block_accesses}
```

## 架构图

```
                    ┌─────────────────────────────────┐
                    │         EngineCore               │
                    │                                  │
  add_request() ──► │  _try_restore_from_cpu_cache()  │ ◄── 恢复路径（两者共用）
                    │         │                        │
                    │         ▼                        │
                    │  ┌─ NPU prefix cache hit? ──┐   │
                    │  │ Yes: 直接使用，不拷贝      │   │
                    │  │ No:  查 CPU cache ────────┤   │
                    │  │      hit → CPU→NPU restore│   │
                    │  │      miss → 正常 prefill   │   │
                    │  └──────────────────────────┘   │
                    │                                  │
  schedule() ─────► │  被动卸载：淘汰时 NPU→CPU       │
  release_kv() ───► │  主动卸载：API 调用时 NPU→CPU   │
                    │         │                        │
                    │         ▼                        │
                    │  collective_rpc                   │
                    │  ("offload_release_blocks")       │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │         Worker (NPU/GPU)         │
                    │                                  │
                    │  NPU KV Cache ◄──swap──► CPU     │
                    │  (显存)        blocks    (pinned │
                    │                         memory)  │
                    └─────────────────────────────────┘
```
