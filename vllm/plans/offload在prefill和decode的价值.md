在当前 PD 架构下，**价值有限**。原因：

```
当前流程（即使 decode 有 CPU cache）：

新请求 → Proxy → Prefill（重新计算 KV）→ Mooncake 传输 → Decode
                    ↑                          ↑
              没省掉这一步               也没省掉这一步
```

Prefill 节点不知道 decode 侧有 CPU 缓存，所以仍然会：
1. **重新计算** KV cache（prefill 算力浪费）
2. **重新传输** 到 decode（Mooncake 带宽浪费）

decode 侧即使从 CPU 恢复了 block，也会被 Mooncake 传来的数据覆盖或冲突。

**真正有价值的方案有三个层次**：

| 方案 | 效果 | 省掉什么 |
|------|------|---------|
| A. **Prefill 侧做 CPU offload** | prefill 检测到 cache hit 后跳过计算 | 省 prefill 计算 + Mooncake 传输 |
| B. **Proxy 感知 decode 的缓存状态** | 直接把请求路由到 decode，跳过 prefill | 省 prefill 计算 + Mooncake 传输 |
| C. **decode 侧通知 prefill 跳过** | decode 告知 connector 已有这些 block | 省 Mooncake 传输（不省 prefill） |

**建议**：把 CPU offload 的重心放到 **prefill 节点**。prefill 侧有完整的 block_hash→KV 映射，release 后 offload 到 CPU，下次相同前缀的请求直接从 CPU 恢复到 NPU，跳过重新计算，Mooncake 也不需要传输。要调整方向吗？


是的，已经有效了。两个节点都配了 `--num-release-cpu-blocks 5000`，proxy 的 `release_kv_cache` 也转发到了两个节点（我们看到 prefill 返回了 `block_released: 9`）。

Prefill 节点的完整链路：

```
1. 首次请求 → prefill 计算 KV → blocks 缓存在 prefix cache
2. release_kv_cache → 9 blocks aged + offloaded 到 CPU  ✅ 已实现
3. 新请求(相同 cache_salt) → add_request 
   → _try_restore_from_cpu_cache 
   → CPU→NPU 恢复 
   → blocks 注册到 prefix cache               ✅ 已实现
4. schedule() → get_computed_blocks 
   → 命中已恢复的 blocks 
   → 跳过这些 tokens 的 prefill 计算           ✅ 已实现
5. 只对剩余 tokens 做 prefill → Mooncake 传输到 decode
```

**所以 prefill 节点才是这个特性的主要受益方**——省掉了重复前缀的计算。可以跑 demo 验证 prefill 节点的日志，应该能看到 `CPU cache hit` 和 `Restored N blocks` 的打印。decode 节点的 CPU offload 可以考虑关掉，避免浪费内存。