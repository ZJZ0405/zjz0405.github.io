---
title: "llm-compressor：vLLM 生态的量化压缩流水线"
published: 2026-11-01T04:10:00+08:00
description: "llm-compressor 是 vLLM 项目的 PTQ 压缩库：用 recipe 组合 GPTQ/AWQ/SmoothQuant/FP8/NVFP4，产出 compressed-tensors checkpoint 直接交给 vLLM。本文梳理两个入口、常见配方、校准集、显存与部署细节。"
tags: [llm-compressor, vLLM, 模型压缩, 量化, 工具链]
category: Quantization
draft: false
---

量化论文给你一个算法；工程落地还需要一串动作：读 Hugging Face checkpoint、准备校准集、逐层统计激活或 Hessian、把 scale 和 zero-point 写进权重、生成推理引擎认识的 `config.json`。llm-compressor 做的就是这层胶水。

它是 vLLM 项目维护的 PTQ 库（Red Hat AI 是主要贡献方），定位很明确：**在 Hugging Face 模型和 vLLM 之间，完成量化与序列化**。它不发明新算法，也不实现推理 kernel；它把 GPTQ、AWQ、SmoothQuant、FP8、NVFP4 这些方法收进同一套 recipe，最后保存成 [compressed-tensors](https://github.com/vllm-project/compressed-tensors) 格式，vLLM 直接加载。

> **五分钟版**
>
> - **两个入口**：`oneshot` 走 Hugging Face 模型，按算法需要跑校准，支持 GPTQ/AWQ/SmoothQuant/REAP；`model_free_ptq` 直接改 safetensors，只做 data-free 方案，适合新架构和超大 checkpoint。
> - **一个 recipe**：modifier 列表决定“用什么算法、量化哪些层、忽略哪些层”；换方案通常只是换 `scheme`。
> - **产物是标准格式**：`compressed-tensors` 的 safetensors + `config.json`，vLLM 自动识别，不需要额外转换。
> - **硬件决定收益**：FP8 要 Ada/Hopper，NVFP4/MXFP4 的激活加速要 Blackwell；老卡能加载，但通常退化成 weight-only。
> - **版本耦合很紧**：0.15.0 对应 torch 2.10–2.14.1、transformers 5.15–5.21、compressed-tensors 0.20.1，生产环境别随意混装。

## 它负责哪一层

把链路拆开看，职责其实很清楚：

| 层 | 谁负责 | 做什么 |
| :--- | :--- | :--- |
| 算法 | llm-compressor 的 modifier | 跑校准、算 scale、更新或重排权重 |
| 序列化 | compressed-tensors | 把量化参数写进 safetensors 和 `config.json` |
| kernel / 推理 | vLLM | 读量化配置，选择对应的 GEMM、Marlin 或 FP4 kernel |

所以 llm-compressor 既不是训练框架，也不是推理引擎。它少数会碰到“训练”的地方是 QAD（quantization-aware distillation）一类 recipe 组件：在权重量化之后，用校准输入和未量化子图的输出做轻量蒸馏，但仍然服务于量化，而不是通用微调。

这也解释了它的边界：模型压完能不能真的变快，取决于 vLLM 里有没有对应 scheme 的 kernel、你的 GPU 支不支持这种数值格式。llm-compressor 保证的是“压得出来、存得对、引擎认”。

## 版本与安装

截至 2026 年 11 月，PyPI 上的稳定版是 `0.15.0`（2026 年 10 月底发布）。安装本身只有一行：

```bash
pip install llmcompressor
python -c "import llmcompressor; print(llmcompressor.__version__)"
```

但依赖窗口值得单独看一眼。`0.15.0` 要求 Python 3.11+，并且把版本卡得比较死：

- `torch >=2.10.0, <=2.14.1`
- `transformers >=5.15.0, <=5.21.0`
- `compressed-tensors ==0.20.1`

这不是维护者偷懒。量化会改写模型结构、拦截 `save_pretrained`、依赖 transformers 的内部模块命名；上游接口一变，整条链路就可能断。生产环境建议把这三个包和 vLLM 一起锁版本，而不是只锁 `llmcompressor`。

官方推荐 Linux + NVIDIA GPU。没有 GPU 也能跑一部分 data-free 的 RTN / FP8 流程，但 GPTQ、AWQ 的校准基本要在 GPU 上做，否则时间不可接受。

## 两个入口，三种职责

llm-compressor 对外的主入口其实只有两个，另有一个格式转换工具：

| | `oneshot` | `model_free_ptq` |
| :--- | :--- | :--- |
| 是否加载 Hugging Face 模型定义 | 是 | 否 |
| 是否使用校准数据 | 算法需要时使用 | 否，只做 data-free 方案 |
| 支持 GPTQ / AWQ / SmoothQuant / REAP | 是 | 否 |
| 支持 FP8 / NVFP4 等 data-free 方案 | 是 | 是 |
| 适合的场景 | 大多数 Hugging Face 模型 | 新架构、超大 checkpoint、`oneshot` 失败时 |

第三个是 `compressed-tensors` 提供的 `convert_checkpoint`：把 ModelOpt NVFP4、AutoAWQ 之类的格式转成 compressed-tensors，或者把压缩 checkpoint 还原成稠密权重再重新量化。它不加载模型，直接在 safetensors 上操作，适合做格式迁移。

`oneshot` 的生命周期分三段：

1. **预处理**：加载模型和 tokenizer/processor，拆开共享的输入/输出 embedding，给 `save_pretrained` 打补丁；
2. **校准**：初始化 recipe 里的 modifier，跑校准前向，让每个 modifier 收集统计量；
3. **后处理**：finalize modifier，把压缩后的权重、tokenizer、recipe 和 config 写进 `output_dir`。

recipe 可以是一个 modifier、一个 modifier 列表，或者一个 YAML 文件。每个 modifier 都有自己的 `initialize` / `calibrate` / `finalize`，所以“先 AWQ 调 scale、再量化”这类组合可以写进同一个列表里。

## 第一个配方：FP8_BLOCK

如果模型在 Ada/Hopper 上，FP8 是最省事的起点：RTN 不需要校准数据，精度通常也够。下面这个例子把线性层压到 FP8_BLOCK，并保留 `lm_head` 和 MoE 的 router gate：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-30B-A3B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

oneshot(
    model=model,
    recipe=recipe,
    output_dir="Qwen3-30B-A3B-FP8-BLOCK",
)
```

几个参数值得展开：

- `targets="Linear"` 表示所有 `torch.nn.Linear`。也可以写模块名或带 `re:` 前缀的正则，只量化某几层。
- `ignore` 里的 `lm_head` 通常保留 16-bit；MoE 的 gate/router 对精度敏感，也建议排除。正则要加 `re:` 前缀，否则会被当成字面量。
- `FP8_BLOCK` 的权重按 128×128 分块量化，激活在推理时按 128 的 group 动态量化。RTN 不需要校准集；如果要静态激活量化，才需要 `oneshot(dataset=...)`。
- `oneshot(output_dir=...)` 会自动保存模型、tokenizer、recipe 和量化配置，`save_compressed` 默认为 `True`。如果传 `output_dir=None`，就只量化不落盘。

压完之后直接交给 vLLM：

```bash
vllm serve Qwen3-30B-A3B-FP8-BLOCK
```

**硬件注意**：FP8 的权重和激活都加速，需要 Ada/Hopper（compute capability ≥ 8.9）。在更老的卡上，checkpoint 仍能加载，但 vLLM 会退回 weight-only 的 Marlin 一类 kernel，激活保持 16-bit；这时你省的是显存，不是算力。

## 4-bit 权重：GPTQ 与 AWQ

4-bit 权重量化是部署默认选项，但 RTN 在 INT4 上经常掉点，需要 GPTQ 或 AWQ 用校准数据找回精度。下面是一份 GPTQ 的完整流程：

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 512 条指令数据，按模型训练时的 chat template 格式化。
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")
ds = ds.shuffle(seed=42)
ds = ds.map(
    lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
)

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    output_dir="Meta-Llama-3-8B-Instruct-W4A16-G128",
)
```

几个经验值：

- **样本数**：官方从 512 起步，掉点明显再加；不是越多越好，校准集太大只会拖长时间。
- **序列长度**：2048 是常见起点；如果线上输入更长，校准长度也应跟着调整。
- **数据分布**：用与部署相近的数据。指令模型就用指令数据，并套它的 chat template；拿通用语料校准指令模型，长尾指令上的退化会更明显。
- **`W4A16` vs `W4A16_ASYM`**：前者对称、无 zero-point，kernel 更简单；后者多一个 zero-point，低比特下通常更稳，但反量化略慢。

`GPTQModifier` 的默认值是 `block_size=128`、`dampening_frac=0.01`、`actorder="static"`。显存紧张时优先降 `block_size` 或改用 `pipeline="sequential"`，不要一上来就把模型切到 CPU 上跑。

目前的 AWQ API 已经拆成“先调 scale、再量化”：

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier

recipe = [
    AWQModifier(),
    QuantizationModifier(
        targets="Linear",
        scheme="W4A16_ASYM",
        ignore=["lm_head"],
    ),
]
```

`AWQModifier` 只负责按激活统计调整权重 scale；真正的量化由后面的 `QuantizationModifier` 完成。它依赖模型家族的 mapping（哪些 norm 对应哪些投影），Llama 一类有内置表；换模型时如果 mapping 缺失，需要自己传 `mappings`。

## 方案怎么选：先看硬件，再看精度

llm-compressor 的 scheme 很多，但选择顺序其实很固定：**先确定 GPU 能加速什么，再在同档位里挑精度最高的**。

| scheme | 精度 | 加速目标 | vLLM 加速要求（最低 CC） | 典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| `W4A16` / `W8A16` | 4/8-bit 权重，16-bit 激活 | 权重 | 7.5 | 显存受限、老卡；decode 收益大 |
| `W8A8` INT8 | 8-bit 整数权重和激活 | 权重 + 激活 | 7.5 | 老卡上的高吞吐 |
| `W8A8` FP8 | 8-bit 浮点权重和激活 | 权重 + 激活 | 8.9 | Ada/Hopper 的通用选择 |
| `NVFP4` | 4-bit 浮点权重和激活 | 权重 + 激活 | 10.0 | Blackwell 上最大压缩 |
| `MXFP4` | 4-bit MX 浮点 | 权重 + 激活 | 10.0 | 跨平台 MX 格式；MoE 常用 |
| `MXFP4xFP8` | MXFP4 权重 + FP8 激活 | 权重 + 激活 | 9.0 | Hopper 上的 MoE 专家层 |
| `MXFP8` | 8-bit MX 浮点 | 权重 + 激活 | 10.0 | 比 MXFP4 更保精度 |
| `W4AFP8` | 4-bit 权重 + FP8 激活 | 权重 + 激活 | 9.0 | 低比特权重配 FP8 激活 |
| `W4AINT8` | 4-bit 权重 + INT8 激活 | 权重 + 激活 | —（Arm） | Arm 平台 |

这张表里的 compute capability 是“权重和激活都加速”的要求。NVFP4、MXFP4、MXFP8 的 checkpoint 在 Ampere 及更新的卡上也能加载，但 vLLM 会改用 weight-only 的 Marlin kernel，激活保持 16-bit。**能加载不等于能加速**，选型时要把这一点算进去。

另外两类方案不在这张表里，但对长上下文很关键：

- **KV cache 量化**：`kv_cache_scheme` 可以把 K/V 压到 FP8；对 32K 以上的上下文，KV cache 往往比权重更占显存。
- **Attention 量化**：实验性的 NVFP4 attention 已经能跑，但需要确认 vLLM 版本和 kernel 支持。

## 算法谱系：什么时候值得多花时间

| 算法 | 机制 | 什么时候用 | 代价 |
| :--- | :--- | :--- | :--- |
| RTN | 就近舍入到目标精度 | FP8/FP4 的快速基线；先跑通流程 | 4-bit 整数上容易掉点 |
| GPTQ | 逐层二阶（Hessian）误差补偿 | 4-bit 权重、需要恢复精度 | 需要校准；显存和时间开销较大 |
| AWQ | 按激活重要性缩放权重 | 4-bit 权重，尤其指令模型 | 需要校准；依赖模型 mapping |
| SmoothQuant | 把激活 outlier 迁移到权重 | W8A8，激活难量化时 | 校准便宜，常与 GPTQ/AWQ 组合 |
| AutoRound | 符号梯度下降优化 rounding/clipping | 4-bit、希望比 RTN 更稳 | 需要额外依赖 `auto-round`，较慢 |
| SpinQuant / QuIP | 旋转或不相干变换压低 outlier | W4A4 等极低比特 | 变换要 fuse，否则白做 |
| REAP | 按 router 加权激活显著性剪掉 MoE 专家 | MoE 显存不够 | 需要校准；剪多了会伤领域能力 |

再往上看，还有两类“组合拳”：

- **HIGGS 混合精度**：收集逐层 MSE 敏感度，用整数线性规划在目标平均位宽下分配各层精度。它解决的是“哪些层该 4-bit、哪些该 8-bit”，比手工拍脑袋靠谱。
- **QAD**：在权重量化之后跑量化感知蒸馏，用未量化子图的输出当 teacher，不额外加载一个 teacher 模型。适合 PTQ 掉点、又不想做完整 QAT 的场景。

需要特别说明的是，**2:4 稀疏压缩已经从 llm-compressor 移除**。官方给出的理由是硬件支持和用户需求都不足；现在保留的剪枝入口是 REAP，面向 MoE 专家，不是稠密模型的 2:4 或非结构化稀疏。看到旧教程里的 `SparseGPTModifier`，先确认版本。

## 大模型与工程细节

### 校准流水线

`oneshot` 的 `pipeline` 参数决定校准前向怎么跑：

| pipeline | 做法 | 适合 |
| :--- | :--- | :--- |
| `independent` | 每个 modifier 各自跑前向（默认） | 大多数场景 |
| `sequential` | 逐层校准，层间激活可 offload | 单卡放不下的大模型 |
| `datafree` | 不做前向，只 initialize/finalize | data-free 权重量化 |
| `basic` | 所有 modifier 共享一次前向 | 简单的后处理校准 |

大模型优先试 `sequential`：它一次只把一层放上 GPU，层间激活可以放到 CPU，或者第二张卡（`sequential_offload_device="cuda:1"`）。再大就上 disk offloading 或 DDP；README 里这两条路径都有示例，但都需要额外配置，不是加一个参数就能自动完成。

### MoE 模型

MoE 的坑在于校准数据只路由到少数专家，没被激活的专家统计量是空的。llm-compressor 默认 `moe_calibrate_all_experts=True`，让所有 token 过所有专家；代价是校准变慢、显存变高。如果模型有 MTP 层、共享专家或特殊的 router 结构，最好先看 `examples/quantizing_moe/` 里对应模型的脚本，而不是直接套通用 recipe。

非均匀量化也支持：不同层可以给不同 scheme，例如专家层 NVFP4、attention 和共享专家 FP8。这比“全模型一个精度”更接近生产做法，但 recipe 会复杂不少。

### 没有 Hugging Face 定义的新模型

新架构刚发布、transformers 还没合入时，`oneshot` 会失败。这时用 `model_free_ptq`：

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="meta-llama/Meta-Llama-3-8B-Instruct",
    save_directory="Meta-Llama-3-8B-Instruct-FP8-BLOCK",
    scheme="FP8_BLOCK",
    ignore=["lm_head"],
    device="cuda:0",
)
```

它按 safetensors 分片处理，不需要把完整模型加载成 `torch.nn.Module`；可以传 `device=["cuda:0", "cuda:1"]` 做多卡分片，也可以接 `converter` 做 SpinQuant 旋转或 ModelOpt 格式转换。`ignore` 里以 `norm` 结尾的模块会被自动跳过，这一点比 `oneshot` 省心。

### 保存与兼容性

`oneshot` 保存的是 compressed-tensors 格式，量化信息在 `config.json` 的 `quantization_config` 里。vLLM 启动时会读这个字段，选择对应的 kernel。升级 vLLM 时，如果新版本改了 scheme 名称或 kernel 支持范围，旧 checkpoint 可能仍然能加载，但会走 fallback 路径——**吞吐对不上时，先查 vLLM 的启动日志里实际选了什么 kernel**。

## 怎么判断压完还能不能用

最省事的做法是接 `lm-evaluation-harness`：

```bash
lm_eval --model vllm \
  --model_args pretrained=./Meta-Llama-3-8B-Instruct-W4A16-G128,add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size auto
```

`add_bos_token=true` 不是可选项：量化模型对 BOS 很敏感，lm-eval 默认不加，分数会莫名其妙地低。官方 W4A16 示例在 GSM-8K 5-shot、250 条上限下报 `0.728`，FP8-Dynamic 示例报 `0.768`；这两个数字只说明流程跑通，不是模型排名，别跨模型、跨硬件直接比。

除了任务分数，至少再看三件事：

- **perplexity**：在领域数据上看一眼，确认没有整体崩掉；
- **长上下文**：如果线上有 32K 输入，校准和评测都要覆盖，KV cache 量化尤其要单独测；
- **端到端吞吐**：TTFT、TPOT、显存峰值，和未量化版本在同一 batch/并发下比，而不是只看单条延迟。

## 什么时候选它

适合：

- 部署栈是 **vLLM + NVIDIA GPU**，模型来自 Hugging Face；
- 需要 PTQ，方案在 W4A16、W8A8、FP8、FP4、KV cache 之间；
- 希望产物是标准格式，不自己维护一套量化脚本。

不适合：

- 目标是 **OpenVINO / Intel CPU / iGPU / NPU**：看 [NNCF](/posts/quantization/nncf/)；
- 目标是 TensorRT-LLM 或厂商私有格式：llm-compressor 的产物不是它们的原生输入；
- 需要从零训练或完整 QAT：它只提供 QAD 这类轻量路径，完整训练仍要自己的 pipeline。

和 NNCF 的分工可以用一句话概括：**llm-compressor 把模型压给 vLLM，NNCF 把模型压给 OpenVINO**。两者都做 PTQ 和低比特权重，但目标运行时、后端限制和产物格式不同；同一个模型要同时上 GPU 和 CPU，通常需要各跑一遍。

## 参考

- [llm-compressor 仓库](https://github.com/vllm-project/llm-compressor)
- [llm-compressor 文档](https://docs.vllm.ai/projects/llm-compressor/en/latest/)
- [PyPI: llmcompressor](https://pypi.org/project/llmcompressor/)
- [compressed-tensors](https://github.com/vllm-project/compressed-tensors)
- [vLLM 量化支持](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- 本分类的相关长文：[量化从零开始](/posts/quantization/from-scratch/)、[量化论文汇总](/posts/quantization/papers/)
