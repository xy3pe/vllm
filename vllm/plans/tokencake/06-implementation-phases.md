# 06 — 分阶段实施计划

> 三阶段渐进实施，每个 Phase 独立可用、独立可测。

---

## Phase 1 — 函数调用感知 offload（核心价值，2-3 周）

**目标**：实现 call_start/call_finish → should_offload → Host RAM offload 的完整通路。

### 改动清单

| # | 文件 | 改动描述 | 新增/修改 | 预估行数 |
|---|------|---------|----------|---------|
| 1 | `vllm/v1/request.py` | 新增 STALLED_ON_FC 状态、OffloadStatus 枚举、fc_* 字段 | 修改 | ~60 |
| 2 | `vllm/entrypoints/openai/protocol.py` | 新增 CallStartRequest/Response、CallFinishRequest/Response | 修改 | ~50 |
| 3 | `vllm/entrypoints/openai/api_server.py` | 新增 /v1/agent/call_start、/v1/agent/call_finish 路由 | 修改 | ~40 |
| 4 | `vllm/v1/engine/core.py` | 新增 AgentCallStartEvent/FinishEvent 事件类型、event queue | 修改 | ~60 |
| 5 | `vllm/v1/core/sched/time_scheduler.py` | **新建** TimeScheduler 类、EWMA 模型、should_offload | 新增 | ~250 |
| 6 | `vllm/v1/core/sched/scheduler.py` | 集成 TimeScheduler、STALLED 请求处理、_process_agent_events | 修改 | ~150 |
| 7 | `vllm/config.py` | 新增 tokencake_enabled 开关、TimeSchedulerConfig | 修改 | ~30 |
| **合计** | | | | **~640** |

### 依赖关系

```
[1] Request 状态机 ──┐
[2] Protocol 模型  ──┼──▶ [6] Scheduler 集成 ──▶ 集成测试
[3] API 路由      ──┤
[4] Engine 事件    ──┤
[5] TimeScheduler ──┘
[7] Config ────────────────────────────────────┘
```

### 测试要点

- [ ] call_start 后请求进入 STALLED_ON_FC 状态
- [ ] should_offload=True 时 KV 被 offload 到 Host RAM
- [ ] call_finish 后请求恢复 RUNNING 并继续推理
- [ ] EWMA 预测值随调用次数收敛到真实均值
- [ ] tokencake_enabled=False 时零影响
- [ ] 压力测试：大量并发 call_start/call_finish 不导致死锁

### 验收指标

- 工具调用期间 HBM 利用率下降（blocks 释放）
- 同等 HBM 下支持更多并发请求（throughput 提升）
- call_start/call_finish API 延迟 < 1ms

---

## Phase 2 — Predictive Upload + UCM 集成（1-2 周）

**目标**：实现预测性上传（零等待恢复）和 UCM Store 持久化 offload。

### 改动清单

| # | 文件 | 改动描述 | 新增/修改 | 预估行数 |
|---|------|---------|----------|---------|
| 8 | `vllm/v1/core/sched/time_scheduler.py` | check_predictive_uploads()、gradual reservation | 修改 | ~80 |
| 9 | `vllm/v1/core/sched/offload_router.py` | **新建** OffloadRouter、多层级存储选路 | 新增 | ~80 |
| 10 | `vllm/v1/core/kv_cache_manager.py` | reserve_blocks() 方法（gradual reservation） | 修改 | ~40 |
| 11 | `vllm_ascend/distributed/ucm_connector.py` | prepare_offload/upload Scheduler 侧 | 修改 | ~100 |
| 12 | `ucm/integration/vllm/ucm_connector.py` | execute_offload/upload Worker 侧 | 修改 | ~120 |
| 13 | `vllm/v1/core/sched/scheduler.py` | _check_predictive_uploads、gradual reservation 循环 | 修改 | ~60 |
| **合计** | | | | **~480** |

### 依赖

- 依赖 Phase 1 全部完成
- 依赖 UCM Store 环境可用（NFS / Mooncake 配置）

### 测试要点

- [ ] predictive upload 在 call_finish 前完成（zero-wait resume）
- [ ] gradual reservation 分多轮预留 blocks，无调度抖动
- [ ] OffloadRouter 正确按 predict_time 选路（Host/UCM Local/UCM Remote）
- [ ] UCM Store offload 后可通过 prefix cache 命中（跨请求复用）
- [ ] offload 到 UCM + upload 回 HBM 数据一致性

### 验收指标

- call_finish 后恢复延迟 < 5ms（predictive upload 命中时）
- UCM prefix cache 命中率提升（offload 副产品）

---

## Phase 3 — Space Scheduler + 混合优先级（1 周）

**目标**：DAG 感知的优先级调度和内存分区保护。

### 改动清单

| # | 文件 | 改动描述 | 新增/修改 | 预估行数 |
|---|------|---------|----------|---------|
| 14 | `vllm/v1/core/sched/space_scheduler.py` | **新建** SpaceScheduler、混合优先级、内存分区 | 新增 | ~200 |
| 15 | `vllm/entrypoints/openai/protocol.py` | AgentMetaRequest/Response | 修改 | ~20 |
| 16 | `vllm/entrypoints/openai/api_server.py` | /v1/agent/meta 路由 | 修改 | ~15 |
| 17 | `vllm/v1/core/sched/scheduler.py` | 集成 SpaceScheduler、抢占保护、周期更新 | 修改 | ~80 |
| 18 | `vllm/v1/core/kv_cache_manager.py` | apply_reservations()、分区分配逻辑 | 修改 | ~60 |
| 19 | `vllm/config.py` | SpaceSchedulerConfig | 修改 | ~20 |
| **合计** | | | | **~395** |

### 依赖

- Phase 1（STALLED 状态和 agent_type 字段）
- Phase 2（可选，但 offload 路径可增强分区效果）

### 测试要点

- [ ] 关键 agent 的请求优先获得 KV blocks
- [ ] 非关键 agent 优先被抢占
- [ ] 内存分区随 GPU 利用率动态调整
- [ ] 混合优先级正确融合 static + dynamic 分量
- [ ] 单 agent 场景下（无 agent_type）退化为原始行为

### 验收指标

- 多 Agent 场景端到端延迟降低（关键路径不被阻塞）
- 非关键 agent 无饥饿（dynamic priority 防饥饿）

---

## 总代码量估算

| Phase | 新增 | 修改 | 总行数 | 新增文件 |
|-------|------|------|-------|---------|
| Phase 1 | ~250 | ~390 | ~640 | 1 (time_scheduler.py) |
| Phase 2 | ~80 | ~400 | ~480 | 1 (offload_router.py) |
| Phase 3 | ~200 | ~195 | ~395 | 1 (space_scheduler.py) |
| **合计** | **~530** | **~985** | **~1515** | **3** |

---

## 完整文件影响矩阵

```
                           Phase 1  Phase 2  Phase 3
vllm/v1/request.py            ✦
vllm/entrypoints/.../protocol.py  ✦                ✦
vllm/entrypoints/.../api_server.py ✦               ✦
vllm/v1/engine/core.py        ✦
vllm/v1/core/sched/
  ├── time_scheduler.py        ★        ✦
  ├── offload_router.py                 ★
  ├── space_scheduler.py                         ★
  └── scheduler.py             ✦        ✦        ✦
vllm/v1/core/kv_cache_manager.py        ✦        ✦
vllm/config.py                 ✦                 ✦
vllm_ascend/.../ucm_connector.py        ✦
ucm/.../ucm_connector.py               ✦

★ = 新建文件   ✦ = 修改文件
```

---

## 配置示例

### 启动参数

```bash
# Phase 1 最小配置
vllm serve Qwen/Qwen3-0.6B \
  --tokencake-enabled \
  --tokencake-offload-enabled

# Phase 2 + UCM
vllm serve Qwen/Qwen3-0.6B \
  --tokencake-enabled \
  --tokencake-offload-enabled \
  --kv-connector UCMConnector \
  --kv-connector-extra-config '{"UCM_CONFIG_FILE": "./ucm.yaml"}'

# Phase 3 完整配置
vllm serve Qwen/Qwen3-0.6B \
  --tokencake-enabled \
  --tokencake-offload-enabled \
  --tokencake-priority-enabled \
  --tokencake-w-static 10.0 \
  --tokencake-reserve-ratio 0.2
```

### 环境变量（可选覆盖）

```bash
VLLM_TOKENCAKE_ENABLED=1
VLLM_TOKENCAKE_OFFLOAD_TIME_PER_BLOCK=0.001  # 硬件标定值
VLLM_TOKENCAKE_UPLOAD_TIME_PER_BLOCK=0.001
VLLM_TOKENCAKE_MIN_OFFLOAD_WINDOW=0.5
```

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| offload 中 call_finish 提前到达 | KV 在 Host 未传完，需紧急恢复 | cancel_or_reverse 机制 + 同步 fallback |
| EWMA 预测偏差大 | 错误 offload 导致无谓传输开销 | safety_margin=1.5 + outlier_factor 过滤 |
| 内存分区导致碎片化 | shared pool 不足 | min_reserve_ratio=0 + 动态调整 |
| UCM Store 写入失败 | offload 失败，blocks 仍在 HBM | fallback 到 STALLED+NONE，不影响正确性 |
| 多 TP rank 下 block hash 不一致 | UCM 读取错误数据 | 复用现有 RequestHasher TP rank 重哈希机制 |
| Feature flag 关闭时的性能回归 | 额外的 if 检查开销 | 热路径上单次 bool 检查，ns 级开销 |
