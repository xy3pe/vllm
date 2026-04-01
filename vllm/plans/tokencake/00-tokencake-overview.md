# TokenCake 集成方案总览

> vLLM / vLLM-Ascend + UCMConnector 支持 TokenCake 机制
>
> 上层应用（DeepSearch 等）负责 DAG 定义和 predict_time 填充，
> 本方案仅涉及 vLLM serving engine 侧改动。

---

## 方案目标

在多 Agent 应用中，LLM 推理与外部函数调用交替执行：

```
Agent 请求生命周期：
  [LLM Prefill] → [LLM Decode → tool_call] → [函数调用 3s] → [LLM Decode 继续] → ...
                                                 ↑
                                          KV Cache 空占 HBM
                                          GPU/NPU 空转
```

TokenCake 的核心思想：**函数调用期间主动将 KV Cache 卸载，腾出 HBM 给其他请求。**

---

## 方案文件索引

| 文件 | 内容 |
|------|------|
| `00-tokencake-overview.md` | 本文件 — 总览与架构 |
| `01-api-layer.md` | API 层：call_start / call_finish / agent_meta 端点 |
| `02-request-statemachine.md` | Request 状态机扩展：STALLED_ON_FC 状态 |
| `03-time-scheduler.md` | Time Scheduler：offload 决策 + EWMA 预测 + predictive upload |
| `04-space-scheduler.md` | Space Scheduler：混合优先级 + 内存分区 |
| `05-ucm-integration.md` | UCM 协同：offload 路径复用 + 多层级存储 |
| `06-implementation-phases.md` | 分阶段实施计划与改动文件清单 |

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  上层应用 (DeepSearch / LangGraph / CrewAI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ DAG 定义      │  │ predict_time │  │ call_start / call_finish  │ │
│  │ (应用负责)    │  │ (应用负责)    │  │ (应用在 tool call 时调用) │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬──────────────┘ │
└─────────┼────────────────┼────────────────────────┼────────────────┘
          │                │                        │
          ▼                ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  vLLM API Layer (本方案新增)                                        │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ POST /v1/agent   │  │ POST /call_start│  │ POST /call_finish │  │
│  │   /meta          │  │                 │  │                   │  │
│  │ (注册 agent 元数据)│  │ (通知函数调用开始)│  │ (通知函数调用结束) │  │
│  └────────┬─────────┘  └────────┬───────┘  └────────┬───────────┘  │
└───────────┼─────────────────────┼────────────────────┼─────────────┘
            │                     │                    │
            ▼                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  vLLM Scheduler Process                                             │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │ Time Scheduler   │   │ vLLM Scheduler   │   │ Space Scheduler │  │
│  │ (新增组件)        │   │ (现有，扩展)      │   │ (新增组件)       │  │
│  │                  │   │                  │   │                 │  │
│  │ should_offload() │──▶│ Request 状态机    │◀──│ hybrid_priority │  │
│  │ predict_fc()     │   │ STALLED_ON_FC    │   │ memory_reserve  │  │
│  │ predictive_      │   │ KVCacheManager   │   │                 │  │
│  │   upload()       │   │ PriorityQueue    │   │                 │  │
│  └────────┬─────────┘   └────────┬─────────┘   └─────────────────┘  │
│           │                      │                                   │
│           ▼                      ▼                                   │
│  ┌──────────────────────────────────────────────┐                   │
│  │ UCMConnector (Scheduler 侧)                   │                   │
│  │ build_connector_meta() — 含 offload/upload 指令│                   │
│  └──────────────────────┬───────────────────────┘                   │
└─────────────────────────┼───────────────────────────────────────────┘
                          │ ConnectorMetadata (load/dump/offload specs)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  vLLM Worker Process (Ascend NPU)                                   │
│  ┌──────────────┐  ┌─────────────────────┐  ┌───────────────────┐  │
│  │ Model Runner  │  │ UCMConnector Worker │  │ Offload Worker    │  │
│  │ (NPU Forward) │  │ (layer-wise I/O)    │  │ (HBM↔Host async)  │  │
│  └──────────────┘  └─────────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Memory Hierarchy                                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ NPU HBM       │  │ Host RAM     │  │ UCM Store                │ │
│  │ shared+reserved│  │ block buffer │  │ NFS / Mooncake / Remote  │ │
│  └───────────────┘  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心设计原则

1. **非侵入性** — 所有新增功能通过 feature flag 控制，不影响现有非 Agent 场景
2. **复用优先** — 最大化复用现有 OffloadingManager / UCMConnector 基础设施
3. **上层无感** — 上层应用只需调用 3 个 REST API，无需了解 vLLM 内部调度细节
4. **渐进式** — 分 3 个 Phase 实施，每个 Phase 独立可用

---

## 核心数据流

```
上层应用                    vLLM                           Memory
─────────                 ─────                          ──────

1. /chat/completions ────▶ Scheduler 分配 KV blocks
                           Request 状态 = RUNNING
                           LLM 推理中...

2. LLM 输出 tool_call ───▶ Response 流式返回给应用
   (应用拿到 tool_call)

3. POST /call_start ─────▶ Time Scheduler:
   {request_id,              should_offload()?
    fc_type,                   │
    predict_time}              ├─ Yes → 异步 HBM→Host    ──▶ Host RAM
                               │        blocks 标记           (可选→UCM Store)
                               │        PENDING_OFFLOAD
                               │        Request 状态 =
                               │        STALLED_ON_FC
                               │
                               └─ No  → Request 状态 =
                                        STALLED_ON_FC
                                        (KV 留在 HBM)

4. (函数调用执行中...)        Scheduler 照常调度其他请求
                           利用释放的 HBM blocks

                           当 remaining_time ≈ T_upload:
                           predictive_upload()           ◀── Host RAM→HBM

5. POST /call_finish ────▶ Time Scheduler:
   {request_id,              EWMA 更新
    actual_duration}         Request 状态 = RUNNING
                           紧急 upload (如果还没完成)
                           Scheduler 恢复调度该请求
```
