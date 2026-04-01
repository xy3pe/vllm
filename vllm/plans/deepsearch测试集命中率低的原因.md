**命中率低是正常的。** 原因：

### 1. 请求前缀高度多样化

| 指标 | 数值 |
|------|------|
| 总请求数 | 908 |
| 不同 system prompt (200字符前缀) | **526** |
| 完全不同的 system prompt | **711** |
| 前400字符相同的分组 | 553 组，其中仅 159 组有复用 |

908 个请求中有 711 个完全不同的 system prompt，前缀多样性极高。

### 2. 时间戳导致 hash 不同

很多 system prompt 以 `Current Time: Tue Mar 17 06:33:16 2026 UTC` 开头，时间戳不同就导致前 128 tokens 内容不同 → block hash 不同 → 无法命中。

### 3. 最大复用组也很小

复用最多的前缀也只有 30 个请求共享，且它们的后续内容（user message）往往不同，实际能共享的完整 block 更少。

### 4. block_size=128 进一步降低命中概率

NPU 的 block_size=128 tokens，需要连续 128 个 token 完全一致才能产生一个 block hash。即使 system prompt 相似，只要前 128 tokens 内有任何差异（如时间戳），整个 block 就无法命中。

### 结论

这个测试集是 **deep research agent 的多 agent 调用链**，每个请求承担不同角色（greeter、planner、outliner、searcher、collector...），prompt 本身就高度异构。prefix cache 和 CPU offload 对这类场景效果有限。

**适合 CPU offload 的场景**：同一用户多轮对话（共享长 system prompt + 历史消息前缀），或批量处理相同模板的请求。