# 01 — API 层设计

> 新增 3 个 REST 端点，供上层应用在函数调用生命周期中通知 vLLM。

---

## 1. 端点总览

| 端点 | 方法 | 触发时机 | 核心作用 |
|------|------|---------|---------|
| `/v1/agent/meta` | POST | 请求开始前（可选） | 注册 Agent 元数据（DAG 权重、agent_type） |
| `/v1/agent/call_start` | POST | 应用开始执行 tool call | 触发 offload 决策 |
| `/v1/agent/call_finish` | POST | tool call 执行完毕 | 更新 EWMA，恢复推理 |

---

## 2. 请求/响应模型

### 2.1 Agent 元数据注册

```python
# vllm/entrypoints/openai/protocol.py 新增

class AgentMetaRequest(OpenAIBaseModel):
    """注册 Agent 元信息，影响 Space Scheduler 的优先级计算。"""
    request_id: str                        # 对应 /chat/completions 的 request_id
    agent_type: str                        # agent 类型标识，如 "programmer", "reviewer"
    static_priority: float = 0.0           # DAG 关键路径权重，上层计算后传入
    # 可选：上层可以在每次请求时附带，也可以在 /chat/completions 的 extra_body 中传入

class AgentMetaResponse(OpenAIBaseModel):
    success: bool
    message: str = ""
```

### 2.2 函数调用开始

```python
class CallStartRequest(OpenAIBaseModel):
    """通知 vLLM 某个请求进入函数调用阶段。"""
    request_id: str                        # 正在执行的请求 ID
    fc_type: str                           # 函数调用类型，如 "web_search:tavily"
    predict_time: float | None = None      # 应用侧预估时间（秒），None 则使用冷启动默认值
    num_stages: int = 1                    # 函数调用的子阶段数（用于细粒度进度追踪）
    stage_name: str | None = None          # 当前阶段名（可选）

class CallStartResponse(OpenAIBaseModel):
    success: bool
    offload_decision: bool = False         # 是否决定 offload（信息性，不要求应用处理）
    message: str = ""
```

### 2.3 函数调用结束

```python
class CallFinishRequest(OpenAIBaseModel):
    """通知 vLLM 函数调用已完成，请求可恢复推理。"""
    request_id: str
    actual_duration: float                 # 实际耗时（秒）
    stage_name: str | None = None          # 完成的阶段名（可选）
    error: bool = False                    # 函数调用是否失败

class CallFinishResponse(OpenAIBaseModel):
    success: bool
    upload_status: str = "not_needed"      # "completed" | "in_progress" | "not_needed"
    message: str = ""
```

---

## 3. 替代方案：通过 extra_body 内联传递

对于不想单独调用 API 的场景，也可以在 `/chat/completions` 的 `extra_body` 中传入 agent 元数据：

```json
{
    "model": "Qwen3-0.6B",
    "messages": [...],
    "extra_body": {
        "agent_type": "programmer",
        "static_priority": 0.85
    }
}
```

这样 Space Scheduler 可以在请求到达时就获得优先级信息，无需单独调用 `/v1/agent/meta`。

---

## 4. API Server 注册方式

在 `vllm/entrypoints/openai/api_server.py` 中新增路由：

```python
# ==================== TokenCake Agent API ====================

@router.post("/v1/agent/meta")
async def agent_meta(request: AgentMetaRequest, raw_request: Request):
    """Register agent metadata for Space Scheduler priority."""
    client = engine_client(raw_request)
    result = await client.agent_meta(
        request_id=request.request_id,
        agent_type=request.agent_type,
        static_priority=request.static_priority,
    )
    return JSONResponse(content=AgentMetaResponse(
        success=result.success,
        message=result.message,
    ).model_dump())


@router.post("/v1/agent/call_start")
async def call_start(request: CallStartRequest, raw_request: Request):
    """Notify engine that a request has entered function call phase."""
    client = engine_client(raw_request)
    result = await client.call_start(
        request_id=request.request_id,
        fc_type=request.fc_type,
        predict_time=request.predict_time,
        num_stages=request.num_stages,
    )
    return JSONResponse(content=CallStartResponse(
        success=result.success,
        offload_decision=result.offload_decision,
        message=result.message,
    ).model_dump())


@router.post("/v1/agent/call_finish")
async def call_finish(request: CallFinishRequest, raw_request: Request):
    """Notify engine that function call has completed, resume inference."""
    client = engine_client(raw_request)
    result = await client.call_finish(
        request_id=request.request_id,
        actual_duration=request.actual_duration,
        error=request.error,
    )
    return JSONResponse(content=CallFinishResponse(
        success=result.success,
        upload_status=result.upload_status,
        message=result.message,
    ).model_dump())
```

---

## 5. EngineCore 事件传递

API 端点通过 `EngineClient` → `EngineCoreClient` → `EngineCore` 链路将事件传递到 Scheduler。

### 5.1 新增 EngineCoreRequest 类型

```python
# vllm/v1/engine/core.py 或 vllm/v1/engine/protocol.py 中新增

@dataclass
class AgentCallStartEvent:
    request_id: str
    fc_type: str
    predict_time: float | None
    timestamp: float               # 事件到达时间

@dataclass
class AgentCallFinishEvent:
    request_id: str
    actual_duration: float
    error: bool
    timestamp: float

@dataclass
class AgentMetaEvent:
    request_id: str
    agent_type: str
    static_priority: float
```

### 5.2 Scheduler 处理入口

在 `Scheduler.schedule()` 的调度循环开始前，处理待处理的 Agent 事件队列：

```python
# scheduler.py — schedule() 方法开头新增
def schedule(self) -> SchedulerOutput:
    # ---- TokenCake: 处理 agent 事件 ----
    self._process_agent_events()

    # ---- 现有调度逻辑 ----
    ...

def _process_agent_events(self):
    while self._agent_event_queue:
        event = self._agent_event_queue.popleft()
        if isinstance(event, AgentCallStartEvent):
            self._handle_call_start(event)
        elif isinstance(event, AgentCallFinishEvent):
            self._handle_call_finish(event)
        elif isinstance(event, AgentMetaEvent):
            self._handle_agent_meta(event)
```

---

## 6. Feature Flag

所有 Agent API 功能通过配置开关控制：

```python
# vllm/config.py — SchedulerConfig 扩展
class SchedulerConfig:
    ...
    tokencake_enabled: bool = False           # 总开关
    tokencake_offload_enabled: bool = True    # Time Scheduler offload
    tokencake_priority_enabled: bool = True   # Space Scheduler 优先级
```

当 `tokencake_enabled=False` 时：
- 3 个端点正常返回 `success=True` 但不执行任何逻辑（静默接受）
- 不影响现有调度行为
- 零额外开销

---

## 7. 上层应用调用示例

```python
import httpx

VLLM_URL = "http://localhost:8000"

# 1. 发起推理请求
resp = httpx.post(f"{VLLM_URL}/v1/chat/completions", json={
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "搜索 Python 性能优化"}],
    "extra_body": {
        "agent_type": "researcher",
        "static_priority": 0.85,
    }
})
request_id = resp.json()["id"]

# 2. 应用解析到 tool_call，开始执行搜索
httpx.post(f"{VLLM_URL}/v1/agent/call_start", json={
    "request_id": request_id,
    "fc_type": "web_search:tavily",
    "predict_time": 3.0,
})

# 3. 执行实际搜索
start = time.time()
search_result = tavily_search("Python 性能优化")
duration = time.time() - start

# 4. 搜索完成，通知 vLLM 恢复
httpx.post(f"{VLLM_URL}/v1/agent/call_finish", json={
    "request_id": request_id,
    "actual_duration": duration,
})

# 5. 继续推理（将搜索结果作为新 message 发送）
resp = httpx.post(f"{VLLM_URL}/v1/chat/completions", json={
    "model": "Qwen3-0.6B",
    "messages": [..., {"role": "tool", "content": search_result}],
})
```
