---
title: "NNCF：OpenVINO 生态的模型压缩框架"
published: 2026-11-01T04:20:00+08:00
description: "NNCF 是 Intel 的 Neural Network Compression Framework，也是 OpenVINO 的官方压缩后端：PTQ、INT4/INT8 权重压缩、AWQ/GPTQ/Scale Estimation、QAT、剪枝与精度控制。本文按使用场景梳理 API、参数、后端限制与部署链路。"
tags: [NNCF, OpenVINO, 模型压缩, 量化, 工具链]
category: Quantization
draft: false
---

如果部署目标是 Intel CPU、iGPU 或 NPU，链路通常是 PyTorch / Hugging Face → OpenVINO IR → OpenVINO Runtime 或 GenAI。NNCF（Neural Network Compression Framework）就是这条链路上的压缩后端。

它和 [llm-compressor](/posts/quantization/llm-compressor/) 经常被放在一起比较：两者都做 PTQ、都支持 INT4/INT8 权重压缩、都有 AWQ/GPTQ 一类校准算法。但目标运行时不同——llm-compressor 把模型压给 vLLM，NNCF 把模型压给 OpenVINO。选哪个，首先取决于模型最后跑在哪里。

NNCF 的范围也比“LLM 量化库”宽：卷积网络、检测/分割、语音、Transformer 都能用；对 LLM 的核心入口是 `compress_weights`，对一般模型是 `quantize`。它还保留 QAT、通用剪枝和精度感知量化，llm-compressor 目前没有对应的完整路径。

> **五分钟版**
>
> - **三种主要用法**：`nncf.quantize` 做 INT8 PTQ；`nncf.compress_weights` 做 LLM 权重量化（INT8/INT4/FP8/NVFP4）；`nncf.quantize_with_accuracy_control` 在掉点时自动回退敏感层。
> - **权重压缩的关键参数**：`mode`、`group_size`、`ratio`、`backup_mode`；默认会把 embedding、卷积和最后一层留在 8-bit，`all_layers=True` 才会一起压到 4-bit。
> - **数据感知更强但更慢**：给 `dataset` 后可以用 AWQ、Scale Estimation、GPTQ、LoRA Correction 找激活 outlier；官方在 Xeon Gold 6430L 上测到比 data-free 平均慢约 30%。
> - **后端限制很硬**：NF4、MXFP4、MXFP8、NVFP4、FP8 这些低比特浮点模式的支持因后端而异；PyTorch/TorchFX 只覆盖一部分，GPTQ/LoRA 也不在 PyTorch 后端。
> - **版本要配对**：3.4.0 对应 OpenVINO 2026.4.0、PyTorch 2.13.0、ONNX 1.22.0；3.x 起不再有 TensorFlow 后端。

## 定位：OpenVINO 的压缩后端，不是推理引擎

NNCF 由 Intel 维护，Apache 2.0。它本身不做推理，而是把压缩算法作用到模型图上：

| 层 | 谁负责 | 做什么 |
| :--- | :--- | :--- |
| 压缩算法 | NNCF | 插量化层、算 scale、选混合精度、剪枝、QAT |
| 图表示 / 运行时 | OpenVINO | 加载压缩后的 IR，调度 CPU/iGPU/NPU kernel |
| 前端集成 | Optimum Intel、Ultralytics 等 | 把 NNCF 接进导出和训练流水线 |

它支持的后端和状态大致如下：

| 后端 | 状态 | 说明 |
| :--- | :--- | :--- |
| OpenVINO | 首选 | PTQ、权重压缩、精度控制最完整 |
| PyTorch | 支持 | PTQ、QAT、剪枝；权重压缩只支持部分模式 |
| TorchFX | 实验 | 走 `torch.export` 的图；部分功能缺失 |
| ONNX | 支持 | PTQ、权重压缩；低比特模式支持有限 |
| TensorFlow | 3.x 移除 | 老项目需停在 2.19，或先转 ONNX |

被谁用过这件事，比自我介绍更能说明定位：Optimum Intel 用它做 `optimum-cli export openvino` 的量化后端，Ultralytics 用它量化 OpenVINO 导出模型，ExecuTorch 的 OpenVINO 集成、`torch.compile` 的 OpenVINO 后端、OpenVINO Training Extensions 和 Microsoft Olive 也都把它当压缩引擎。

## 安装与版本

```bash
pip install nncf
pip install openvino          # OpenVINO 后端
pip install "nncf[plots]"     # 需要画敏感度/精度图时
```

NNCF 不会替你装后端。PyPI 包只依赖通用库（numpy、scipy、scikit-learn、safetensors 等），实际用哪个框架要自己装；`import nncf` 时如果一个后端都找不到，会提示没有可用后端。

Python 要求是 3.10+，3.4.0 的官方测试组合是 Python 3.12。版本矩阵建议贴在环境里：

| NNCF | OpenVINO | PyTorch | ONNX | Python |
| :--- | :--- | :--- | :--- | :--- |
| 3.4.0 | 2026.4.0 | 2.13.0 | 1.22.0 | 3.12 |
| 3.3.0 | 2026.3.0 | 2.10.0 | 1.22.0 | 3.12 |
| 2.19.0 | 2025.4.0 | 2.8.0 | 1.17.0 | 3.10（最后一个带 TF 的版本线） |

NNCF 的 API 相对稳定，但压缩后的模型格式、支持的 op 和 OpenVINO 版本强相关。用 Optimum Intel 时，NNCF 版本通常由 `optimum[openvino]` 间接决定；不要手工把 NNCF 升到比 OpenVINO 新一档，然后再去查为什么模型加载失败。

## API 地图：先分清“量化激活”和“只压权重”

NNCF 的入口不少，但按用途分只有几类：

| API | 做什么 | 典型后端 |
| :--- | :--- | :--- |
| `nncf.quantize` | INT8 PTQ，权重和激活都量化 | OpenVINO、PyTorch、ONNX、TorchFX |
| `nncf.compress_weights` | 只压权重，激活保持浮点 | OpenVINO、PyTorch、ONNX、TorchFX |
| `nncf.quantize_with_accuracy_control` | PTQ + 精度回退，掉点超阈值时把敏感层留在高精度 | OpenVINO、ONNX |
| `nncf.prune` | 非结构化剪枝（magnitude / 正则化） | PyTorch |
| `nncf.batch_norm_adaptation` | 剪枝后重估 BN 统计量 | PyTorch |
| QAT | `quantize` 后直接 fine-tune | PyTorch |

对 LLM 来说，最常用的其实是 `compress_weights`：它只动权重，不需要改 attention 的数值路径，部署侧只要 OpenVINO 能反量化即可。`quantize` 更适合 CNN、检测、分割这类对激活量化不敏感、又需要 INT8 吞吐的模型。

## PTQ：`nncf.quantize`

PTQ 的流程很短：给模型、给一小份校准集、调用一次 API。

```python
import nncf
import openvino as ov

model = ov.Core().read_model("model.xml")

def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(val_loader, transform_fn)

quantized_model = nncf.quantize(
    model,
    calibration_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    target_device=nncf.TargetDevice.CPU,
    subset_size=300,
    ignored_scope=nncf.Scope(types=["Multiply", "Subtract"]),
)

ov.save_model(quantized_model, "model_int8.xml")
```

参数不多，但每个都影响结果：

- **`preset`**：`PERFORMANCE` 对权重和激活都用对称量化；`MIXED` 对权重对称、对激活非对称。默认值会根据 `model_type` 推断：Transformer 用 `MIXED`，其他用 `PERFORMANCE`。
- **`target_device`**：`CPU`、`GPU`、`NPU` 或 `ANY`。它影响 NNCF 选择的量化方案，不是运行时设备选择。
- **`subset_size`**：默认 300 条。校准集太小，激活统计不稳；太大则收益递减。
- **`model_type=nncf.ModelType.TRANSFORMER`**：让 NNCF 用 Transformer 的 pattern 处理注意力和 MLP，LLM/Transformer 上建议显式设置。
- **`ignored_scope`**：按名字、正则、op 类型或子图排除节点。`nncf.IgnoredScope` 现在是 `nncf.Scope` 的兼容别名，新代码直接用 `Scope`。
- **`fast_bias_correction`**：默认 `True`，用 FastBiasCorrection 修正量化前后的偏差；设为 `False` 会换成更准但更慢、更省内存的版本。

`transform_fn` 的返回值跟后端有关：OpenVINO、PyTorch、TorchFX 直接返回模型输入张量；ONNX 要返回 `{input_name: np.ndarray}`。如果模型有多个输入（LLM 的 `input_ids`、`attention_mask`、`position_ids`、`beam_idx`），返回 dict。

一个容易踩的限制：**Transformer 类模型不要用 batch size > 1 的 dataloader 做校准**。NNCF 文档明确说，这类模型的 batch 轴位置不固定，大于 1 时统计会不准；`subset_size` 也要按 `subset_size // batch_size` 重算。

## LLM 权重压缩：`nncf.compress_weights`

这是 NNCF 在 LLM 上最常用的入口。最小用法：

```python
from nncf import compress_weights, CompressWeightsMode

compressed_model = compress_weights(
    model,
    mode=CompressWeightsMode.INT4_SYM,
    group_size=128,
    ratio=0.8,
)
```

`ratio=0.8` 的含义是：在“非特殊层”里，按敏感度把 80% 的权重压到 4-bit，其余留在 backup 精度。对 INT4/INT8 主模式，默认 backup 是 `INT8_ASYM`；embedding、卷积和最后一层默认不参与 4-bit，除非设 `all_layers=True`。

### 模式怎么选

| 模式 | 元素类型 | scale | 粒度 | 主要后端 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `INT8_ASYM` | INT8 | FP16 | per-channel | 全部 | 默认，精度最稳，相对 FP32 约 4× |
| `INT8_SYM` | INT8 | FP16 | per-channel | 全部 | 无 zero-point，kernel 更简单 |
| `INT4_SYM` | INT4 | FP16 | per-channel / group | 全部 | 最快，但可能掉点明显 |
| `INT4_ASYM` | INT4 | FP16 | per-channel / group | 全部 | 比对称稳，反量化略慢 |
| `NF4` | FP32 查找表 | FP16 | per-channel / group | OpenVINO | 16 个正态分位值；精度好，但不一定更快 |
| `MXFP4` | E2M1 | E8M0 | group=32 | OpenVINO | OCP MX 格式，跨平台 |
| `MXFP8_E4M3` | E4M3 | E8M0 | group=32 | OpenVINO | MX FP8，精度高于 MXFP4 |
| `FP8_E4M3` | E4M3 | FP16 | per-channel / group | OpenVINO、ONNX | 常规 FP8 |
| `FP4` | E2M1 | FP16 | per-channel / group | OpenVINO | 非 MX 的 FP4 |
| `NVFP4` | E2M1 | E4M3 per group + FP32 per weight | group=16 | OpenVINO | NVIDIA 的 FP4 布局 |
| `CB4` / `CODEBOOK` | 4-bit 码本 | FP16 | per-channel / group | OpenVINO | 固定/自定义查找表 |

源码里还有 `INT3_SYM`、`INT2_SYM`、`ADAPTIVE_CODEBOOK` 等模式，但文档没有把它们列为主推路径；生产使用前先确认目标 OpenVINO 版本和 kernel 是否真的支持。**模式支持矩阵比模式数量重要**——NF4、MXFP4、MXFP8、NVFP4 这类低比特浮点模式目前只有 OpenVINO 后端完整支持；ONNX 支持 `FP8_E4M3`，PyTorch/TorchFX 连 FP8 也不支持。选之前先看后端。

### 三个最常调的参数

- **`group_size`**：多少权重共享一个 scale。`128` 是 LLM 的常见起点；`32`/`64` 更准但 scale 更多、反量化更慢；`-1` 表示 per-channel，不再分组。4-bit 下通常 `64` 或 `128`。
- **`ratio`**：4-bit 层占比。`0.8` 表示 80% 的（非特殊）层压到 4-bit，剩下 20% 留 8-bit。想要更小就调高，掉点就调低。
- **`backup_mode`**：混合精度里“高精度那档”用什么。对 INT4/INT8 主模式默认是 `INT8_ASYM`；也可以设 `NONE`（保留 FP16/BF16）、`INT8_SYM`、`FP8_E4M3`、`MXFP8_E4M3`。`all_layers=True` 会把 embedding 和最后一层也按主精度压，通常不建议一上来就开。

### 数据感知：AWQ、Scale Estimation、GPTQ、LoRA Correction

只给 `mode` 和 `ratio` 时，NNCF 用权重本身估计敏感度（data-free）。如果给一份校准集，它可以用激活统计找 outlier，把更敏感的层留在高精度；再打开 `awq=True`、`scale_estimation=True`、`gptq=True`、`lora_correction=True`，还能进一步修正权重：

```python
compressed_model = compress_weights(
    model,
    mode=CompressWeightsMode.INT4_SYM,
    group_size=128,
    ratio=0.8,
    dataset=nncf_dataset,
    awq=True,
    scale_estimation=True,
)
```

这几个开关的组合关系需要记一下：

- **AWQ + SE**、**AWQ + LC**、**SE + LC**、**AWQ + SE + LC** 可以任意组合；
- **GPTQ** 可以和 AWQ、SE 组合；
- **GPTQ 与 LoRA Correction 不能同时开**，源码里直接报错；
- **SE、GPTQ、LoRA Correction 必须提供 `dataset`**；
- 这些数据感知算法目前主要在 **OpenVINO 后端** 上用，且不支持 MXFP4/MXFP8/FP8/FP4/NVFP4 主模式。

LoRA Correction 会插入额外的线性层来补偿量化噪声，精度收益通常最好，但会带来内存和运行时开销；如果部署侧对峰值内存敏感，先用 SE 或 GPTQ。

### 给特定层单独配置

MoE 模型是典型场景：专家层可以压到 4-bit，router 最好留在 8-bit。`scoped_configs` 可以按名字、正则、op 类型或子图覆盖全局配置：

```python
import nncf
from nncf import compress_weights, CompressWeightsMode, ScopedConfig

compressed_model = compress_weights(
    model,
    mode=CompressWeightsMode.INT4_SYM,
    group_size=64,
    ratio=1.0,
    scoped_configs=[
        ScopedConfig(
            patterns=[".*self_attn.*", ".*router.*"],
            mode=CompressWeightsMode.INT8_ASYM,
            group_size=-1,
        ),
    ],
)
```

匹配规则和 `nncf.Scope` 一致；多个 `ScopedConfig` 命中同一层时，**列表中靠后的优先**。`ratio` 只统计没有被 `scoped_configs` 或 `ignored_scope` 覆盖的层，这一点和“全局 80%”的直觉不完全一样。

### 缓存统计量

调 `group_size` / `ratio` 时，每次重新跑校准很浪费时间。把统计量缓存下来：

```python
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters

compressed_model = compress_weights(
    model,
    mode=CompressWeightsMode.INT4_SYM,
    group_size=128,
    advanced_parameters=AdvancedCompressionParameters(statistics_path="statistics"),
)
```

第一次会计算并写入 `statistics`，之后直接读。注意这个功能只在 OpenVINO 后端可用；Torch/TorchFX/ONNX 会报“不支持 statistics caching”。

### 在 OpenVINO LLM 上怎么接

如果模型是 `optimum.intel.openvino.OVModelForCausalLM`，压缩的是它内部的 `ov.Model`：

```python
from functools import partial

import numpy as np
import nncf
from datasets import load_dataset
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = OVModelForCausalLM.from_pretrained(
    MODEL_ID, export=True, load_in_8bit=False, compile=False
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

def transform_fn(item, tokenizer):
    tokens = tokenizer(item["text"], return_tensors="np")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": np.arange(input_ids.shape[0], dtype=int),
    }

nncf_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

model.model = nncf.compress_weights(
    model.model,
    dataset=nncf_dataset,
    mode=nncf.CompressWeightsMode.INT4_SYM,
    ratio=0.8,
    sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
)
model.save_pretrained("tinyllama_int4")
```

这份代码对应 NNCF 仓库里的 TinyLlama 示例。几个点：

- `sensitivity_metric` 决定混合精度怎么排：`HESSIAN_INPUT_ACTIVATION` 通常比只看权重误差更准，但更慢；`MEAN_ACTIVATION_MAGNITUDE` 更快。
- `OVModelForCausalLM.save_pretrained` 会把压缩后的 IR、tokenizer 和 config 一起存下来，之后可以直接 `from_pretrained` 加载。
- 如果只想要 `ov.Model`，不经过 Optimum，就直接 `compress_weights(ov_model, ...)`，再用 `ov.save_model`。

## 精度控制：掉点之后怎么办

PTQ 掉点超过预期时，NNCF 提供两条路：**精度感知量化**和**QAT**。

### 精度感知量化

`quantize_with_accuracy_control` 会先量化，再用验证集评估；如果掉点超过 `max_drop`，它会把最敏感的层回退到浮点，直到满足约束或没有可回退的层：

```python
quantized_model = nncf.quantize_with_accuracy_control(
    model,
    calibration_dataset,
    validation_dataset,
    validation_fn=validation_fn,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE,
)
```

`validation_fn(model, dataset)` 要返回一个“越大越好”的指标；`drop_type` 可以是绝对值 `ABSOLUTE` 或相对值 `RELATIVE`。这个 API 目前只支持 OpenVINO 和 ONNX 后端，PyTorch 不在支持列表里。

它的代价是多次评估，时间比普通 PTQ 长；好处是不用手工维护 `ignored_scope`，适合检测、分割这类“掉 1 个点就不能接受”的模型。

### QAT

PyTorch 后端的 QAT 路径很直接：先 `nncf.quantize` 得到带量化模块的模型，然后按普通训练流程 fine-tune。

```python
quantized_model = nncf.quantize(model, calibration_dataset)

# ... 正常的训练循环，量化参数会一起更新 ...

checkpoint = {
    "state_dict": quantized_model.state_dict(),
    "nncf_config": nncf.torch.get_config(quantized_model),
}
torch.save(checkpoint, "qat.pt")
```

加载时先恢复 NNCF 的模块结构，再灌 state dict：

```python
checkpoint = torch.load("qat.pt")
quantized_model = nncf.torch.load_from_config(model, checkpoint["nncf_config"])
quantized_model.load_state_dict(checkpoint["state_dict"])
```

QAT 文档里有两个容易忽略的点：训练时关掉 Dropout 一类随机层；导出前用 `ov.convert_model(quantized_model.cpu(), example_input=dummy_input)` 转成 OpenVINO IR。LLM 上还有 Weight-Only QAT with LoRA/NLS 的路径，对应 `CompressionFormat.FQ_LORA` / `FQ_LORA_NLS`，适合“PTQ 掉点、完整 QAT 又太贵”的场景。

### 剪枝

`nncf.prune` 目前只在 PyTorch 后端可用，支持三种模式：

- `UNSTRUCTURED_MAGNITUDE_LOCAL`：每个张量内部按绝对值剪；
- `UNSTRUCTURED_MAGNITUDE_GLOBAL`：全局统一阈值；
- `UNSTRUCTURED_REGULARIZATION_BASED`：用 $L_0$ 正则训练，让重要度自己学出来。

```python
pruned_model = nncf.prune(
    model,
    mode=nncf.PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL,
    ratio=0.7,
    examples_inputs=example_input,
)

pruned_model = nncf.batch_norm_adaptation(
    pruned_model,
    calibration_dataset=calibration_dataset,
    num_iterations=200,
)
```

magnitude 剪枝后建议做 BN adaptation，再 fine-tune 几个 epoch；正则化剪枝需要较长的训练 schedule，不适合一次性脚本。剪枝得到的是非结构化稀疏，OpenVINO 不一定能把它变成实际加速——**稀疏率不等于吞吐**，部署前要用目标硬件测。

## 部署链路

压缩完只是开始，最后要落到 OpenVINO：

- **IR**：`ov.save_model(compressed_model, "model.xml")`，权重以 INT8/INT4 等压缩精度连同 scale 一并写进 IR。
- **Optimum 模型**：`model.save_pretrained(dir)`，目录里包含 OpenVINO 的 `.xml/.bin`、tokenizer 和 `config.json`。
- **命令行**：`optimum-cli export openvino -m meta-llama/Llama-3.2-3B-Instruct --weight-format int4 ./Llama-3.2-3B-Instruct-int4`，底层就是 NNCF 的权重压缩；数据感知模式可以加 `--dataset wikitext2`。
- **运行时**：OpenVINO Runtime 直接加载 IR；LLM 场景用 OpenVINO GenAI 的 `LLMPipeline`，或者 Optimum 的 `OVModelForCausalLM`。

`compression_format` 决定权重怎么存：默认 `DQ`（dequantize，反量化后计算），适合 PTQ；`FQ`（fake quantize）保留量化模拟层，给 QAT 用；`FQ_LORA` / `FQ_LORA_NLS` 会额外带 LoRA 修正层。选错格式的典型症状是“模型能加载，但精度和预期对不上”。

## 和 llm-compressor 怎么分工

| | llm-compressor | NNCF |
| :--- | :--- | :--- |
| 维护方 | vLLM 项目 / Red Hat AI | Intel |
| 目标运行时 | vLLM（NVIDIA GPU 为主） | OpenVINO（Intel CPU / iGPU / NPU） |
| 主入口 | `oneshot`、`model_free_ptq` | `quantize`、`compress_weights` |
| 产物 | compressed-tensors safetensors | OpenVINO IR / 压缩后的 PyTorch / ONNX |
| LLM 强项 | GPTQ/AWQ/SmoothQuant/FP8/NVFP4、KV cache | INT4/INT8 权重压缩、AWQ/SE/GPTQ、QAT |
| 非 LLM 模型 | 不是重点 | CNN、检测、分割、语音都覆盖 |
| 训练时压缩 | QAD 等轻量路径 | QAT、LoRA/NLS、剪枝 |
| 版本耦合 | torch / transformers / compressed-tensors | OpenVINO / PyTorch / ONNX |

一句话：**GPU 上跑 vLLM，选 llm-compressor；Intel 平台上跑 OpenVINO，选 NNCF**。同一个模型要同时上两边，通常得各压一份；两者的 checkpoint 格式不通用，不要指望压完直接互相加载。

## 常见坑

- **先确认后端支持的模式**：NF4、MXFP4、NVFP4 在 PyTorch/ONNX 后端不支持，FP8 只在 ONNX 和 OpenVINO 上支持；`compress_weights` 会在入口直接抛 `ParameterNotSupportedError`。
- **特殊层默认不压**：embedding、卷积、最后一层默认走 backup（通常是 INT8_ASYM）。只改 `mode` 不会把它们变成 4-bit，要 `all_layers=True`。
- **`ratio` 不是全局比例**：它只统计未被 `scoped_configs` / `ignored_scope` 覆盖的层，而且特殊层本来就不参与。
- **NF4 精度好但不一定快**：NNCF 文档明确说，NF4 模型目前不应期待比 INT8 非对称更快；它是精度兜底，不是性能方案。
- **INT4_SYM 最快，也最容易掉点**：先试 `INT4_SYM`，掉点就换 `INT4_ASYM`、加 `group_size=64`、降 `ratio`，或者开 AWQ/SE。
- **校准集要像线上数据**：LLM 用 wikitext 校准代码模型，长尾能力会退化；指令模型用指令数据，别拿纯网页文本凑数。
- **Transformer 校准 batch 用 1**：batch > 1 时统计不可靠，`subset_size` 也要按 batch 重算。
- **OpenVINO 版本要对齐**：NNCF 3.4.0 配 OpenVINO 2026.4.0；混装新旧版本时，先看 `nncf` 和 `openvino` 的 release notes。
- **稀疏不等于加速**：非结构化剪枝在通用硬件上很难变成实际吞吐，除非后端有对应的 sparse kernel。

## 参考

- [NNCF 仓库](https://github.com/openvinotoolkit/nncf)
- [NNCF 文档（OpenVINO）](https://docs.openvino.ai/nncf)
- [NNCF API 文档](https://openvinotoolkit.github.io/nncf/autoapi/nncf/)
- [PyPI: nncf](https://pypi.org/project/nncf/)
- [Optimum Intel 的 OpenVINO 优化](https://huggingface.co/docs/optimum-intel/openvino/optimization)
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- 对照阅读：[llm-compressor：vLLM 生态的量化压缩流水线](/posts/quantization/llm-compressor/)
