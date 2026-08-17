# Unified Memory V2 设计：token-based / req-based 双副本方案

> 状态：设计草稿（RFC）
> 作者：Hugo
> 日期：2026-08-16

## 1. 背景与动机

当前 `--enable-unified-memory`（Unified Memory V1，见 `python/sglang/srt/mem_cache/unified_memory_pool.py`）把 KV 和混合状态塞进**一块字节缓冲**，用 `torch.as_strided` 拆成**恰好 2 个子池**（一个 grow-up、一个 grow-down），并在 `MultiEndedAllocator` 上做虚拟→物理页映射。

V1 有两个根本限制：

1. **只支持"两个子池"**。`UnifiedKVPool.__init__` 硬断言 `len(sub_pool_specs) == 2`（`unified_memory_pool.py:233`），且只实现了 MHA/MLA 全注意力 KV + Mamba 状态、以及 full + SWA 两种组合。
2. **DSV4 被显式排除**。DSV4 的池是异构的：SWA ring + C4 压缩 KV（FP8 + lightning indexer）+ C128 压缩 KV + 每层 compress-state，塞不进"两个子池"模型，`kv_cache_configurator.py:392` 用 `and not is_deepseek_v4(...)` 直接挡掉。

更深层的问题：**当前把"请求态（req-based）"的两类职责耦合在了同一份存储里**——既是前缀匹配的规范副本，又是逐 step 被 forward kernel 读写的计算副本。这两类副本的生命周期、稳定性、写频率完全不同，耦合导致：

- 匹配用副本被逐 step 改动，污染了 radix 树的内容寻址；
- 计算副本的逐 step 改写拖累了统一池的压缩（compaction）和 v2p 表；
- 无法对"匹配副本"做独立压缩（如 int8 checkpoint）而不影响计算精度。

V2 的核心思路：**把 KV cache 分成 token-based 与 req-based 两类；req-based 在统一池里只放"仅用于前缀匹配"的规范副本，另外单独放一份类似 DSV4 现有 SWA ring 的"计算副本"。**

## 2. 核心分类学（taxonomy）

### 2.1 token-based（按 token 位置寻址）

- **寻址**：`(req, 位置 pos)`，一行 = 一个 token 的 K/V（或压缩 token）。
- **内容稳定（content-stable）**：一旦写入不再变化，天然可内容寻址 → 进 radix tree。
- **粒度**：page（`--page-size`）为最小单位，radix 匹配按 page 对齐。
- **分类规则**：只包含**服务 full attention 或压缩 attention** 的逐 token canonical；
  由于**没有一份 cache 同时服务 full 与 SW 注意力**，SWA 的 cache 不归 token-based（归 req-based，见 §2.2）。
- 代表：full attention KV（MHA/GQA/MLA，若模型有）、CA4/CA128 压缩 KV、lightning indexer KV。
- 若模型无 full attention（如 DSV4），Region A 只装压缩 KV + indexer。

### 2.2 req-based（按请求寻址）

- **统一抽象**：一切 req-based = **有界滑动窗口的运行态**，统一参数 `sws`（sliding window size），
  用 ring 实现：`slot = req * ring_size + pos % ring_size`，`ring_size = sws × (1 + 投机裕量)`。
- **分类规则**：窗口有界（内核只读最近 `sws` 个位置 / 下一状态只依赖当前状态）→ req-based。
- **内容**：作为 checkpoint 时内容稳定（可进 radix tree）；作为运行态时每 step 被改写（不进 radix tree）。
- 成员（全部视为 SWA）：

  | cache | sws | ring_size（现状） |
  |--|--|--|
  | 滑窗 attention（MLA/GQA/MHA） | W（如 128） | 128（+spec） |
  | DSV4 C4 压缩态 | 8（2·4） | 8/16 |
  | DSV4 C128 压缩态 | 128 | 128/256 |
  | conv | conv_width（如 4） | conv_width |
  | linear-attn | 1 | 1 |

> 关键观察：**同一份 req-based 数据有两种角色**——
> - **前缀匹配角色（canonical）**：在 checkpoint 边界上固定一份，内容寻址，进 radix tree，随树的 LRU 淘汰。
> - **计算角色（working/compute）**：逐 step 被 forward kernel 读改写，随请求释放。

> 关键观察：**同一份 req-based 数据有两种角色**——
> - **前缀匹配角色（canonical）**：在 checkpoint 边界上固定一份，内容寻址，进 radix tree，随树的 LRU 淘汰。
> - **计算角色（working/compute）**：逐 step 被 forward kernel 读改写，随请求释放。

## 3. V2 设计：双副本原则

### 3.1 规范副本（canonical，仅用于前缀匹配）

- **位置**：统一内存池（V2 缓冲）内的 req-based 区域。
- **内容寻址**：进 radix tree，树节点在 checkpoint 边界上引用/持有 canonical slot。
- **写入时机**：prefill 完成一个 chunk / 请求结束时，把当时的状态固化为 checkpoint。
- **写后只读**：一旦进树不可变，被锁住时不被淘汰，LRU 淘汰。
- **可以独立压缩**：canonical 可以 int8/fp8 量化以扩大容量（匹配精度不受影响，见 §5.3）。
- 对应现有概念：`MambaComponent.mamba_value`、`--enable-int8-mamba-checkpoint`、DSV4 的 checkpoint。

### 3.2 计算副本（compute，仅用于计算）

- **位置**：统一池**之外**的每请求 ring/scratch 缓冲，按需分配。
- **模板就是 DSV4 现在的 SWA ring**：`DeepSeekV4UnifiedKVPool` 的 `[0, swa_pages)` 行，按
  `req_pool_indices * swa_window + pos % swa_window` 寻址（`deepseek_v4_memory_pool.py:398-404`）。
- **每 step 读写**：decode/compress 每步改写；不参与 radix 匹配、不做内容寻址。
- **命中重建**：前缀命中时把 canonical 物化进 compute ring（只补缺的尾部窗口），类似
  `swa_reprefill_tail_tokens()`（`unified_radix_cache.py:2084-2099`）强制重灌尾部窗口、
  `free_swa_out_of_window_slots()`（`mem_cache/common.py:47-102`）维护窗口前沿。
- **随请求释放**。

### 3.3 一句话

> **canonical 回答"前缀 P 是否存在、从哪分支"（radix match）；compute 回答"当前请求此刻的状态是什么"（forward kernel）。**

DSV4 的 SWA 已经隐式实现了"计算副本"：ring 是 compute，命中时把尾部窗口重新 prefill 进 ring
（模式 b 的雏形）。V2 把这套机制**泛化到所有 req-based 状态**（滑窗 attention、conv、DSV4 压缩态、
linear-attn），并为每个 req-based 状态补上 Region B 的 canonical checkpoint（模式 a）。

## 4. 缓冲布局（Unified Memory V2）

`UnifiedKVPool` V2 仍是**一块原始 `uint8` 缓冲**，但逻辑上分成两个区域（不再强制 2 个子池）：

```
┌─────────────────────────────────────────────────────────────┐
│ Region A: token-based canonical（页粒度，MultiEndedAllocator v2p）│
│   ├─ token-based 块（模型层序，每层按 ratio 缩放行数）           │
│   │    Full（MHA/GQA/MLA，若有）· CA4(+indexer) · CA128        │
├─────────────────────────────────────────────────────────────┤
│ Region B: req-based canonical（checkpoint 索引，间隔 CI）        │
│   ├─ SWA 窗口快照 [p-W+1, p]（模式 a 才存）                     │
│   ├─ C4/C128 压缩态 checkpoint（按模型裁剪）                    │
│   ├─ conv checkpoint（若模型有）                              │
│   └─ linear-attn checkpoint（若模型有）                        │
└─────────────────────────────────────────────────────────────┘
        ▲ canonical 只服务前缀匹配
compute ring / scratch（统一池之外，逐请求）:
   SWA ring（DSV4 现有）、mamba ping-pong（现有 extra buffer）、DSV4 压缩态运行环
```

### 4.1 Region A：token-based 块（已定稿）

**块定义**：一个 token-based 块 = 1 个 page = `P` 个源 token。块内按**类型 → 层序**（type-major、
layer-minor）放置 canonical 行：先按类型分组（`Full → CA4 → CA128`），类型内按层序 `L0→L_{n-1}`，
每层行数按该层 ratio 缩放。与 Region B checkpoint 的排序规则一致。

| 层类型 | ratio | 每块行数 | 行内容 |
|--|--|--|--|
| Full（MHA/GQA） | 1 | `P` | K + V（GQA = kv_heads 更少，行更小） |
| Full（MLA） | 1 | `P` | 单 latent（nope + rope + scale 打包） |
| CA4 | 4 | `P/4`（latent）+ `P/4`（indexer） | 压缩 KV + lightning indexer KV |
| CA128 | 128 | `P/128` | 压缩 KV（无 indexer） |

**对齐要求**：`P = k · lcm(所有 ratio ∪ {1})`。MHA/GQA/MLA 都是 ratio-1，不改变 lcm；
已发布 DSV4 的 ratio 集 {1,4,128} → lcm=128，CUDA 默认 `P=256`、NPU `P=128` 均满足。

**排序说明（type-major）**：同一类型的层在块内连续排列，`as_strided` 按类型/层的偏移分割各层视图。
一个模型只有一种 full-attention 家族（MHA/GQA 或 MLA），不会混；type-major 使 MLA 层天然连续，
直接满足 trtllm/cutlass 的 dense `.view(-1, P, kv_cache_dim)` 需求。

**算子兼容性**：

| 家族 | 后端 | 需要的视图形态 | type-major 块兼容 |
|--|--|--|--|
| GQA/MHA | flashinfer / fa3 / triton | 页主序 strided `(num_pages, P, H, D)` | ✅ |
| CSA（C4/C128/indexer） | dsv4 Triton 后端 | 逐层 strided paged | ✅ |
| MLA（真实 attention 层） | trtllm / cutlass / flashmla | dense 连续 `.view(-1, P, kv_dim)` | ⚠️ 仅当 Full 类型全为 MLA 时 ✅；与 MHA/GQA 混合则不连续 |

- **澄清**：DSV4 没有 MLA attention 层，只有 SWA + CSA；它的压缩 KV 只是**存储格式长得像
  MLA-latent**（`qk_nope_head_dim FP8 + qk_rope_head_dim BF16 + scale` 单行），由 **dsv4 后端
  自己的 kernel**（`fused_store_cache` 等）逐层 strided 读取，与 trtllm/cutlass/flashmla 无关。
- **风险（保留）**：type-major 只保证"同一类型层连续"。**若 Full 类型内 MLA 与其它 attention
  （MHA/GQA）混合**，按层序排布后 MLA 层仍被夹在中间、`as_strided` 切不出连续段 → dense 系
  MLA 算子无法消费。出路：① 把 Full 拆成 Full-MLA / Full-MHA 两个子类型；② 开发 strided-view
  MLA 算子（见 §9）。

**公式**：

```
block_payload_bytes = Σ_L (P / ratio(L)) × row_bytes(L)
block_page_bytes    = align_up(block_payload_bytes, block_alignment)
total_bytes         = num_blocks × block_page_bytes（+ 尾部 pad，供最末层 dense view 越界读）
num_blocks          = ⌈max_total_tokens / P⌉

itemsize = L.dtype.itemsize  # 当前 view 的 dtype 元素字节数；stride/storage_offset 的单位是元素而非字节

view(L) = as_strided(
    raw.view(L.dtype),
    size           = (num_blocks, P/ratio(L), *L.row_shape),
    stride         = (block_page_bytes/itemsize, row_bytes(L)/itemsize, *row_strides),
    storage_offset = cum_offset(L) / itemsize,     # 按类型→层序累积（同类型层连续，跨类型偏移）
)
```

`block_alignment` 覆盖块内各 dtype / kernel 的对齐要求；各类型起始 offset 也分别按其 dtype
对齐。由于 C4/C128 等压缩行只按 ratio 占据部分源 token，`block_page_bytes / P` **不保证为整数，
也不需要有"每 token 字节数"的物理含义**。Region A 的分配和 compaction 单位是整个
`block_page_bytes`，不是 `P × entry_bytes_per_token`。

**token → 槽映射**：

```
block        = t // P
pos_in_block = (t // ratio(L)) % (P / ratio(L))
```

P 是各 ratio 的公倍数，保证压缩 token 严格落在块内、不跨块。

**例子（P=256，类型·层：Full×1 → CA4×2 → CA128×2）**

行字节（示意）：Full K=64B/V=64B；CA4 latent=32B、indexer=16B；CA128=32B。

| 类型·层 | 行数 | 行字节 | 块内偏移(B) |
|--|--|--|--|
| Full·K | 256 | 64 | 0 |
| Full·V | 256 | 64 | 16384 |
| CA4·L0 latent | 64 | 32 | 32768 |
| CA4·L0 indexer | 64 | 16 | 34816 |
| CA4·L1 latent | 64 | 32 | 35840 |
| CA4·L1 indexer | 64 | 16 | 37888 |
| CA128·L0 | 2 | 32 | 38912 |
| CA128·L1 | 2 | 32 | 38976 |
| **block_bytes** | | | **39040** |

**实现**：泛化 `build_page_major_mha_views`（`layout/page_major.py:39`）为
`build_page_major_block_views(raw, layer_specs, P, num_blocks)`，`layer_specs` 按**类型→层序**
给出 `{type, ratio, row_shape, row_bytes, dtype}`；块为 uint8，各层区域按自己的 dtype view
（DSV4 压缩行是 FP8+BF16+scale 打包的 uint8 行，584B/token）。

### 4.2 Region B：req-based checkpoint（已定稿）

**两个新参数**：

| 参数 | 取值 | 语义 |
|--|--|--|
| `req_checkpoint_mode` | `a`（默认）/ `b` | a：Region B 存 SWA 窗口快照（命中免重算窗口）；b：不存 SWA KV（命中重算窗口） |
| `checkpoint-interval`（CI） | token 数 | Region B 存 checkpoint 的间隔 |

**CI 约束**：

```
CI % page_size == 0          # 对齐 page/块（与 Region A 一致，radix 页对齐）
CI > max_swa_sliding_size    # 见下
```

**为什么 `CI > max_swa_sliding_size`**：
1. 每个 checkpoint 边界都能装下一个**完整**的 SWA 窗口（窗口不跨越两个 checkpoint）；
2. 相邻 checkpoint 的窗口快照**不重叠** → Region B 的 SWA 内存 ≈ O(N)（每 token 只存一次），不随 W 膨胀。

**布局**：`k = p / CI` 只是某条 radix 路径上的 checkpoint ordinal，用于判断边界、深度与策略，
**不是 Region B 的全局 slot id**。同一深度的不同前缀拥有不同状态，其内容 identity 是对应的
radix node；每个节点通过 allocator 获得任意稳定 virtual slot，slot 内存该边界处
**该模型拥有的** req-based canonical，内部按"类型 → 层"两级排序拼接
（type-major、layer-minor，与 Region A 块的纯层序不同）：

```
radix node at checkpoint k（位置 p = k·CI）:
  virtual_slot = RegionBAllocator.alloc(1)
  checkpoint_bytes = Σ_类型 Σ_{层∈类型} size(类型, 层)
┌──────────────────────────────────────────────┐
│ 类型 SWA（模式 a 才存）:                         │
│   SWA.L0 窗口快照 [p-W+1, p]                  │
│   SWA.L1 窗口快照                             │
│   ...                                        │
│ 类型 C4:                                      │
│   C4.L0 压缩态（kv+score ring）                │
│   C4.L1 压缩态                                │
│   ...                                        │
│ 类型 C128:                                    │
│   C128.L0 压缩态                              │
│   ...                                        │
│ 类型 conv（若模型有）:                          │
│   conv.L0 conv 状态                           │
│   ...                                        │
│ 类型 linear（若模型有）:                        │
│   linear.L0 运行态                            │
│   ...                                        │
└──────────────────────────────────────────────┘
size(类型, 层):
  SWA  (模式 a): W × row_bytes_SWA(L)            # 窗口快照 = W 个逐 token KV
  C4:            ring_size(C4) × row_bytes_C4(L) # 压缩态 ring
  C128:          ring_size(C128) × row_bytes_C128(L)
  conv:          conv_state_bytes(L)
  linear:        linear_state_bytes(L)
```

**类型排序**：`SWA → C4 → C128 → conv → linear`（暂定，可按模型配置）。

**DSV4 例子**：page_size=256（CUDA）、max SWA sliding=128 → **CI=256**（最小）。
每 checkpoint = `Σ_C4层 ring_size(8)×row + Σ_C128层 ring_size(128)×row + (模式 a) Σ_SWA层 128×row`；
Region B 容量 = `⌈N/CI⌉` × 每 checkpoint 大小。

**前缀匹配集成**：token-type 与 req-type 分别返回匹配长度：

```
L_token = Region A 最长 page-aligned 命中长度
L_req   = Region B 各 req component 共同认可的可恢复长度

0 <= L_req <= L_token
```

`L_req` 是 `L_token` 以内最新的**实际驻留且通过所有 req component validator** 的 checkpoint，
不是简单的 `floor(L_token/CI)·CI`：Region B 可以独立淘汰，因此最近的 checkpoint 缺失时，
`L_req` 可回退多个 CI，甚至回到 0。由 §7.3 的级联规则保证 Region B 不会比其依赖的
Region A 活得更久，所以只会出现 `L_token >= L_req`。

命中后统一重算 `[L_req, L_token)` 以推进 req compute state；这段 token-type canonical 已命中，
因此只计算、不重新存储（selective-store replay，见 §7.2）。两种模式的差别只在 `L_req` 的
validator 与恢复内容：

- 模式 a：checkpoint 同时恢复 req state 与边界处的 SWA 窗口快照，再 replay 到 `L_token`。
- 模式 b：不存 SWA snapshot；SWA validator 将 `L_req` 回退到可通过正向 replay 重建完整窗口的
  可执行锚点，然后与其它 req state 共用同一条 selective-store replay 路径，不再单设
  `swa_reprefill_tail_tokens` 流程。

**恢复来源**（修正）：C4/C128 压缩态、conv、linear-attn 的状态是**隐状态累积 / 有损压缩的中间态**，
Region A 只有压缩产物、没有逐 token 原始数据 → **无法从 Region A 恢复，必须靠 Region B checkpoint**。
滑窗 attention 在模式 a 下靠 Region B 窗口快照恢复，模式 b 下靠重算。

### 4.3 寻址与分配

- **Region A**：沿用 `MultiEndedAllocator` 虚拟→物理页表，`page_size` 决定 radix 页与块大小。
- **Region B**：slot 粒度，由 checkpoint radix node 持有 allocator 返回的 virtual slot；
  `k=p/CI` 只表示路径深度，不参与物理寻址。完整链路为
  `radix node → virtual slot → v2p[virtual slot] → physical checkpoint bytes`
  （`UnifiedMambaSlotAllocator` 已是这种任意 slot 分配/翻译模式）。
- **compute ring**：独立于统一池，逐请求分配/释放（DSV4 SWA ring 现即如此）。

### 4.4 与 V1 的差异

| | V1 | V2 |
|--|--|--|
| 子池数量 | 硬编码 2 | 逻辑多区域（token 区 / req 区），compute 在池外 |
| req-based 状态 | 单副本，匹配+计算共用 | 双副本：canonical(池内) + compute(池外 ring) |
| 匹配单位 | page / mamba chunk / SWA 窗口（现状） | 不变（见 §6） |
| DSV4 | 排除 | 可纳入（§5） |

## 5. 各 attention 类型的映射

| 类型 | canonical（前缀匹配） | compute（计算） | compute 写频率 | 命中时重建 |
|--|--|--|--|--|
| Full attn（MHA/GQA/MLA） | token-based KV（池内 Region A） | 复用 canonical（只读） | 仅 prefill | 无（内容稳定） |
| 滑窗 attention（SWA） | Region B 窗口快照（模式 a）/ 无（模式 b） | 每请求 SWA ring（池外） | 每 decode step | a：checkpoint 恢复窗口；b：重灌尾部窗口 |
| conv + linear-attn | req-based checkpoint（池内 Region B） | 每请求状态 ring / ping-pong | 每 step（改写） | checkpoint→compute 拷贝 + 补跑 |
| DSV4 C4/C128 压缩态 | req-based checkpoint（池内 Region B） | 每请求状态环（compress-state） | 每产出压缩 token 的 step | checkpoint 恢复（Region A 只有有损压缩产物，无法恢复） |

### 5.1 为什么 DSV4 能纳入 V2

DSV4 的组成恰好都是 V2 的实例：

- **SWA ring** = compute 副本（已有现成模板；无 full attention 层，SWA 的 cache 归 req-based）；
- **压缩 KV（C4/C128 + indexer）** = token-based canonical（Region A）；
- **compress-state（C4/C128 运行态）** = req-based canonical（Region B checkpoint）+ 池外运行环。

`kv_cache_configurator.py:392` 的 `and not is_deepseek_v4(...)` 挡路条件可以移除，
DSV4 走 `init_unified_dsv4_pools`（Region A + Region B 的组装）。

### 5.2 mamba 的双副本

现有 `--enable-mamba-extra-buffer`（ping-pong）已是"计算副本"雏形：两个 slot 乒乓，
一个在写时另一个可捐给 radix（`donate_mamba_ping_pong_slot`）。V2 把它正式化：

- **canonical**：`checkpoint-interval`（CI）边界的固化状态（进树，只读，可 int8）。
- **compute**：ping-pong ring（池外，逐 step 改写）。

### 5.3 精度解耦（canonical 可压缩）

双副本最大的收益之一：**canonical 可以独立于 compute 做压缩**。

- canonical 用 int8/fp8 存（容量翻倍），只服务匹配；
- 命中时把 canonical 反量化/拷贝进 bf16/fp8 compute ring 用于计算。

这泛化了现有的 `--enable-int8-mamba-checkpoint`（`mamba_radix_cache.py:1051` `int8_ckpt_pool`）
和 DSV4 压缩 KV 的 FP8 canonical（`DeepSeekV4SingleKVPool`）。

## 6. 前缀匹配单位（V2 参数化）

匹配粒度由以下参数控制：

| 参数 | 来源 | 作用 |
|--|--|--|
| `--page-size` | `server_args.py:891` | token-based 最小匹配单位，`RadixKey.match` 向下取整到 page |
| `checkpoint-interval`（CI） | 新增（V2），**取代** `mamba_cache_chunk_size` | req-based checkpoint 边界（Region B 存储间隔）；`CI % page_size == 0` 且 `CI > max_swa_sliding_size` |
| `sliding_window_size` | 模型 config（如 DSV4=128） | SWA 类 compute 的门控：命中边界须有 ≥1 窗口连续缓存 |
| `req_checkpoint_mode` | 新增（V2），默认 `a` | SWA canonical 是否存 Region B（a 存窗口快照 / b 命中重算） |

匹配路径依旧：radix walk 逐 token（page 对齐）→ 各组件 validator 门控 `best_match_node`
（`unified_tree_core.py:617-729`）；req-based 的"分支点"落在最近的 checkpoint 边界
`p_i = floor(p/CI)·CI`（取代旧 `mamba_checkpoint_grid` 语义）。

**CI 自动选取**：从约束推导默认值（可被用户覆盖）：

```
CI_min = ceil((max_swa + 1) / page_size) × page_size     # DSV4: ceil(129/256)×256 = 256
```

**权衡**：CI 越大 → Region B 越省内存、写 checkpoint 越少，但命中时重灌 gap 最长 CI。
默认给 `CI_min`（或 `k×CI_min`，`k≥1` 作配额旋钮）。

## 7. 实现要点

### 7.1 组件抽象

新增/泛化 `TreeComponent`：

- `MambaComponent` → 泛化成 `ReqStateComponent`（canonical slot 由树节点持有，按 CI 边界放置）；
- `SWAComponent` 泛化（canonical 走 Region B 窗口快照[模式 a] 或命中重算[模式 b]，compute ring 池外）；
- 新增 `DSV4CompressComponent`（Region B 的压缩态 canonical + 池外运行环，按 CI 边界放置）。

### 7.2 命中重建流程

```
prefix hit
  → radix/component validators 返回 L_token、L_req 与 Region B checkpoint slot
  → scheduler 分配 compute ring，checkpoint → compute（D2D / 反量化 / COW）
  → 构造一次 extend：
      [L_req, L_token)      selective-store replay：计算 hidden、更新 req state，不写 Region A
      [L_token, input_end)  normal prefill：计算并写入新分配的 Region A page
```

**两个起点必须解耦**：

```
compute_start  = L_req
kv_alloc_start = L_token
```

allocator 只为 `[L_token, input_end)` 分配新的 Region A page；`[L_req, L_token)` 的 page table /
`req_to_token` 继续引用已命中的 Region A slot，不能因为参与 replay 而重复分配。

**attention metadata 的 selective-store 语义**：同一个 extend batch 中，token-type store loc
按位置构造：

```
token_store_loc[L_req:L_token]     = INVALID   # 通常 lower 为 slot_mapping=-1
token_store_loc[L_token:input_end] = newly_allocated_slots
req_state_store_loc[...]           = compute_ring_slots
```

因此 replay 仍执行各层 attention / MLP / MoE，产生推进 conv、linear-attn、C4/C128
compress-state 所需的 hidden state，但 Full KV、C4/C128 压缩 KV、indexer 等 token-type
canonical store 被跳过。

`INVALID` 是 metadata 层语义，不把 `-1` 固定成跨 backend ABI：支持负 slot mask 的 kernel
直接使用 `-1`；会把负 loc 翻译到 0 或不支持 mask 的 kernel 使用 reserved sink slot / 独立
store mask / 跳过 store kernel。所有 attention backend 及 DSV4 compressor/indexer store
都需审计；若 backend 的当前 token attention 依赖“先写 cache 再读”，还需改为使用当前 chunk
的 raw K/V 或读取已有 canonical slot。

### 7.3 双区域关联与淘汰规则

**关联机制（树节点 = 双区域关联点）**：radix 树节点同时挂 Region A 与 Region B 的 canonical——
`component_data[FULL/SWA/...]` 存 Region A 槽位，CI 边界节点额外挂 Region B checkpoint 槽位指针
（如同现有 `MambaComponent.mamba_value`）。前缀同一性 = 节点同一性 → 同前缀的逐 token 缓存与
checkpoint 自动落在同一节点，**无需额外"区域间索引表"**；锁也按节点关联（`inc/dec_lock_ref` 沿路径锁所有组件）。

Region B 的 slot identity 与物理地址分离：

```
k = p / CI                                      # 路径上的逻辑 ordinal
virtual_slot = RegionBAllocator.alloc(1)         # 全局唯一的稳定 handle
node.component_data[REQ].value = virtual_slot
physical_slot = RegionBAllocator.translate(virtual_slot)
```

不同 radix 分支即使 `p`、`k` 相同也分配不同 virtual slot。compaction 只更新 v2p，不改树节点保存的
virtual slot。若一个压缩 radix edge 跨过 CI 边界，应在 page-aligned CI 边界拆 node，使 checkpoint、
锁、LRU 与 tombstone 继续复用现有 node/component 生命周期；`CI % page_size == 0` 保证拆分合法。

**淘汰规则（V2）**：

1. **Region A（token-based）：从后往前释放，级联删 Region B**。
   - 走 LRU 叶子淘汰（叶子 = 最长前缀段，等价从尾部往回）；
   - 一段 Region A 被删 → 由它派生的 checkpoint（Region B）**必须一并释放**（结构保证：同一节点一删全删）。
2. **Region B（req-based）：可独立释放，早期 checkpoint 优先**。
   - 命中只消费 `L_token` 以内最新的实际驻留 checkpoint（`L_req`），**更早的 checkpoint 对本次命中无用**；
   - 早期 checkpoint 只服务于"短前缀命中"（重灌便宜），因此**先删早期（低 k）的**，
     保住晚期 checkpoint（长前缀复用能力），提高命中率；
   - 独立于 Region A：删 Region B 不删 Region A（该前缀降级为模式 b 重算）。

由此得到命中不变量：`Region B checkpoint 存在 ⇒ 对应 Region A 前缀仍存在`，所以
`L_req <= L_token`。Region B 独立释放只会扩大 selective-store replay 区间；Region A
释放会级联删除其后的 Region B，不会产生 req-type 匹配比 token-type 更长的状态。

**与现状 sglang 的对照**：

| | 现状 sglang | V2 |
|--|--|--|
| Region A 释放 | LRU 叶子，从后往前，级联删状态 | 同左（规则 1 现状已满足） |
| Region B 可独立释放 | ✅ 组件 LRU / tombstone 内部节点（full 保留） | 同左（规则 2 的"独立性"现状已有） |
| Region B 淘汰顺序 | **LRU**（按状态被消费的 recency，位置无关） | **早期位置优先**（结构性）—— V2 新增 |
| DSV4 压缩态 compute ring | 随 SWA 窗口"早期位置先释放"（`maybe_evict_dsv4_state_on_swa`，`dsv4_common_hooks.py:525`） | canonical 也按位置优先 |

**锁与生命周期**：

- **canonical**：被锁（某请求命中中）不得淘汰；`inc/dec_lock_ref`、`evict_mamba`/压缩态淘汰路径沿用。
- **compute ring**：逐请求，请求结束随 req 释放，不进树、不参与 eviction。
- decode 期间只需锁 compute 覆盖的尾部窗口（`release_window_lock`，`swa_component.py:686`）。

### 7.4 分配器（复用 v2p 算法，重构为 page/block-native）与尺寸

V2 = **Region A（页粒度）+ Region B（slot 粒度）**，正好对应当前 **2-ended MultiEndedAllocator**
模型（一个 grow-up、一个 grow-down，共享一块缓冲）。V2 保留双端增长、虚拟页 ID、v2p/p2v、
free、compaction 和 in-flight 保护，但需要把 allocator 的字节模型从
`entry_bytes_per_page = entry_bytes_per_token × page_size` 改为由 sub-pool **直接给出**
`tokens_per_page` 与 `page_bytes`：

```
SubPoolSpec:
    tokens_per_page  # 逻辑上一个物理页覆盖多少源 token
    page_bytes       # 一个物理页/block 的实际字节跨度

num_physical_pages = floor(pool_byte_capacity / page_bytes)
num_virtual_tokens = num_physical_pages × tokens_per_page
```

- **Region A**：`tokens_per_page=P`、`page_bytes=block_page_bytes`；v2p 映射 block，watermark、
  available size、zero/move/compaction 全部按 `page_bytes` 计。压缩场景下
  `page_bytes/P` 可以不是整数。
- **Region B**：`tokens_per_page=1`、`page_bytes=checkpoint_entry_bytes`，继续复用
  `UnifiedMambaSlotAllocator` 的 slot 分配模式（类位于 `unified_memory_pool.py:856`）。

`UnifiedKVPool` 的 `len(sub_pool_specs)==2` 断言（`unified_memory_pool.py:233`）对 V2 **恰好满足**
（A + B 两个区域）；不需要放宽为 N-ended，但 `SubPoolSpec`、`UnifiedKVPool.max_slots` 和
`MultiEndedAllocator.entry_bytes_per_page` 的容量/地址计算需要 page-native 重构。只有未来要拆
更细子池（如 Region A 内各类型独立 allocator）才需放宽 2 子池。

**尺寸（复用当前 unified 公式）**：当前 `init_unified_mamba_pools`（`unified_memory_pool.py:1163`）：

```
total_bytes = max_total_num_tokens × full_entry_bytes
            + max_mamba_cache_size × mamba_entry_bytes
```

其中 `max_mamba_cache_size` 由 `--mamba-full-memory-ratio` 推导
（`kv_cache_configurator.py:2075-2116`：`mamba_budget = total_rest × ratio/(1+ratio)`）。
两个 sub-pool 在 2-ended allocator 里**动态共享**（不是硬分区）。V2 对应地：

```
total_bytes = RegionA_block_budget × block_page_bytes
            + RegionB_checkpoint_budget × checkpoint_entry_bytes

RegionA_block_budget = ⌈RegionA_token_budget / P⌉
```

Region B 预算 = `⌈N/CI⌉ × 每 checkpoint 字节`（每 checkpoint = §4.2 的 type-major 布局）。

### 7.5 compute ring（参考 DSV4，需泛化）

**模板**（DSV4 已有两套）：
- **SWA ring**（`DeepSeekV4UnifiedKVPool` 的 `[0, swa_pages)` 区）：per-request、
  `req*swa_window + pos%swa_window`、`full_to_swa_index_mapping`、`free_swa_out_of_window_slots` 随窗推进；
- **`CompressStatePool`**（`deepseek_v4_compress_state.py`）：per-request 状态 ring（kv+score），`ring_size` 参数化。

**直接照搬不够的点**：

| 不足 | V2 泛化 |
|--|--|
| 5 类 ring 未合一（SWA ring 与压缩态池独立） | 一个 per-request compute-ring 区域容纳 SWA + C4 + C128 + conv + linear 五类 |
| 槽内容分 token-KV 与状态两种 | 统一 sws 参数化的 ring，槽可为 KV 或状态 |
| ring_size 硬编码 | 统一 `ring_size = sws × (1+投机裕量)` |
| 无 checkpoint 恢复路径（只重灌/重算） | 加 `Region B checkpoint → ring` 的 D2D 恢复（mode a） |
| 无统一 settle 语义 | 见 §7.6 |

**寻址**：per-request `slot(req, type, pos) = req_ring_base(type) + (pos / ratio(type)) % ring_size(type)`；
窗口推进按各类型 sws（把 `free_swa_out_of_window_slots` 泛化为按类型）。

### 7.6 settle / 提交语义

**token-type（Region A）**：块直接使用；按 **settle 状态**判定一个块是否还会被改：
- in-flight（正在被写入）的块 → 不可匹配 / 不可复用 / 不可淘汰（"最后一块是热的"）；
- 写完后 settle → 变为只读 canonical，可匹配、可淘汰。

**req-type（Region B）**：只存 checkpoint：
- 计算中运行态在 compute ring，每 step 更新（in-flight）；
- 按 **position** 判定：到 CI 边界时固化一份到 Region B（checkpoint 写入）；
- 按 **settle 状态**判定存储区的 checkpoint 是否可用（写完后才可被命中读取）。

对应现状：`MultiEndedAllocator._inflight_forward` 跟踪在飞 forward 的写集，compaction 不碰未 settle 的页。

## 8. 迁移路径（分阶段）

1. **Phase 0**：引入类型体系，并把 allocator 重构为 page/block-native：
   `SubPoolSpec` 增加 `is_req_based`、`tokens_per_page`、`page_bytes`；保持 v2p/p2v 与
   2-ended 算法不变，容量、watermark、move/compaction 改为直接按 `page_bytes` 计算；文档落库。
2. **Phase 1（DSV4 打通）**：把 DSV4 SWA ring + 压缩态接入双副本：canonical 进 Region A/B，
   compute ring 独立。→ 解除 DSV4 排除，证明 unified + DSV4。
3. **Phase 2（mamba 重构）**：mamba conv/temporal 拆成 canonical + compute ring，保留 V1 行为为兼容模式。
4. **Phase 3（N 区域 + 压缩解耦）**：放宽 2 子池；canonical 独立量化；对接 HiCache（canonical 为宿主备份单位）。

## 9. 开放问题 / 风险

1. **命中时的物化开销**：canonical → compute 的 D2D 拷贝。缓解：只拷尾部窗口、只读状态 COW、异步流。
2. **配额划分**：Region A vs Region B 的比例（类似 `--mamba-full-memory-ratio` 的旋钮）。尺寸公式见 §7.4；
   未定的是 Region B 预算的具体旋钮语义（直接给 checkpoint 数 vs 给字节比例），以及 compute ring 的总预算
   （DSV4 SWA ring 现在的 `swa_size` 逻辑）。
3. **canonical 量化**：int8 canonical → bf16 compute 需反量化路径（DSV4 FP8 压缩 KV 已有先例）。
4. **HiCache 联动**：canonical 是天然的宿主备份单位（D→H write-through/load-back）；compute ring 设备端专属（同 SWA 现状）。
5. **流式会话 / lock ref**：compute ring 每请求持有，跨轮会话需重建；canonical 锁语义与现有 tree lock 一致。
6. **投机解码（DSPARK/MTP）**：draft token 的状态只出现在 compute ring（DSV4 online C128 MTP），
   canonical 不得包含 draft 状态。
7. **PP/PD/DP**：Region A/B 的逐层切分（PP 已有 `stage_ratios` 切片先例）、PD 整包传输的字节连续约束
   （`get_contiguous_buf_infos`）需在 Region B 上重新审计。
8. **混合模型的新算子**：type-major 只保证"同一类型层连续"。若 **Full 类型内 MLA 与 MHA/GQA 混合**，
   MLA 层仍不连续，dense 系 MLA 算子（trtllm/cutlass/flashmla）无法消费 → 需要把 Full 拆成
   Full-MLA / Full-MHA 子类型，或**开发 strided-view MLA 算子**（见 §4.1 算子兼容性）。
9. **Region B 恢复与 selective-store backend 审计**：命中时需要统一的
   `checkpoint→compute ring` 恢复算子（D2D / 反量化 / COW），并要求 attention metadata
   对 `[L_req,L_token)` 构造 INVALID token store loc。普通 attention、DSV4 compressor/indexer、
   CUDA graph、PP/PD 路径需分别验证 INVALID 是真正 mask、sink write 还是需要跳过 store kernel。

## 10. 术语表

- **canonical（规范副本）**：统一池内、内容寻址、只服务前缀匹配、随树 LRU 淘汰的 req-based 状态。
- **compute ring（计算副本）**：统一池外、逐请求、逐 step 读写的运行缓冲（DSV4 SWA ring 为模板）。
- **checkpoint 边界**：canonical 被固化的位置，粒度为 `checkpoint-interval`（CI）；`CI % page_size == 0` 且 `CI > max_swa_sliding_size`。
- **token-based / req-based**：按"是否内容寻址的逐 token canonical / 是否窗口有界的 per-request 缓存"划分；
  没有 cache 同时服务 full 与 SW 注意力，故 SWA 的 cache 归 req-based。
