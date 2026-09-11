# py-models — SHARP → ONNX（WebGPU 目标）

把 Apple 的 [ml-sharp](https://github.com/apple/ml-sharp)（`SHARP`）导出为可在
`onnxruntime-web` + **WebGPU** 上运行的 ONNX。只保留 5 个脚本，其余探索代码与产物归档在
`out/_obsolete/`（历史说明见 `out/_obsolete/README_full.md`）。

## 脚本

| 脚本 | 用途 |
|---|---|
| `common.py` | 共享工具：路径、fp32 基座导出、`Identity` 折叠、fp16 转换、评估指标 |
| `01_smoke_test.py` | PyTorch 侧冒烟测试（环境 / 权重 / 前向 / 输出规格） |
| `02_export_fp16.py` | 导出 fp16 ONNX —— **可移植目标**（CPU / CUDA / DirectML / CoreML） |
| `03_export_mm4f16.py` | 导出 4bit MatMul 权重 + fp16 其余 —— **WebGPU 目标**（`rtn` / `hqq`） |
| `04_evaluate.py` | 统一评估（体积 / PSNR / 3DGS 语义指标 / WebGPU 覆盖率） |

## 用法

```bash
cd py-models
# 环境：venv（Python 3.14 + torch 2.14 + onnxruntime）；两个 clone 以 --no-deps editable 安装

venv/bin/python scripts/01_smoke_test.py mps      # 冒烟测试（设备可换 cpu/cuda）
venv/bin/python scripts/02_export_fp16.py         # -> out/onnx/sharp_fp16.onnx
venv/bin/python scripts/03_export_mm4f16.py rtn   # -> out/onnx/sharp_mm4f16_{rtn,hqq}_bs{32,128}.onnx
venv/bin/python scripts/04_evaluate.py            # -> out/onnx/eval_report.json
venv/bin/python scripts/04_evaluate.py hqq_bs32   # 只测某个变体 -> eval_report_hqq_bs32.json
```

## 产物（`out/onnx/`）

| 文件 | 说明 |
|---|---|
| `sharp.onnx` (+`.data`) | fp32 基座（2.62 GB，评估基准来源） |
| `sharp_clean.onnx` (+`.data`) | 折叠 `Identity` 后的图，4bit 量化的输入（中间产物，可再生） |
| `sharp_fp16.onnx` (+`.data`) | fp16，可移植目标（1.31 GB） |
| **`sharp_mm4f16_hqq_bs32.onnx`** (+`.data`) | **推荐的 WebGPU 目标（0.48 GB）** |
| `ref_outputs.npz` | fp32 ORT 参考张量与输入（评估基准） |

`sharp.onnx` 与 `sharp_clean.onnx` 权重完全相同，差别只在图结构：原始导出把 `nn.Linear`
权重写成 `MatMul(x, Identity(W))`，量化器认不出这种 B 不是常量 → 必须折叠后才可量化。
fp16 反而用**未折叠**的原始图（折叠会让 `Resize.scales` 变成共享常量而被误转 fp16）。

## 量化参数约定

`03` 中为固定常量：`bits=4`、`is_symmetric=False`、`accuracy_level=2`，
`block_size` 取 `32 / 128` 两档用于对比。`accuracy_level=2` 表示 MatMulNBits 内部把输入 A
降为 fp16 计算，与产物其余部分的 fp16 一致。（HQQ 不支持 symmetric，且不写
`accuracy_level`，脚本事后补 `accuracy_level=2`。）

## 简要结果

单张测试图（`clones/ml-depth-pro/data/example.jpg`，1,179,648 个高斯），基准 = fp32 ORT 输出。
原始数据见 `out/onnx/eval_report.json`，日志 `out/logs/04_eval.log`。

| 变体 | 体积GB | 压缩 | disparity | opacities | colors | 可见色误差<br>(8bit,α加权) | 实心区<br>\|Δα\|中位 | WebGPU |
|---|---|---|---|---|---|---|---|---|
| fp32 | 2.623 | 1.00× | — | — | — | — | — | 99.79% |
| **fp16** | 1.313 | 2.00× | **72.73** | **38.47** | 55.19 | **0.270** | **0.0002** | 99.79% |
| **hqq_bs32** | 0.482 | **5.44×** | 41.44 | 26.92 | 39.11 | 1.282 | 0.0004 | 99.79% |

* **`block_size=32` 优于 128**：体积多约 10%，但 disparity PSNR 高 5~7 dB、可见颜色误差低
  35~40%（量化粒度更细）；同粒度下 `hqq ≥ rtn`。
* **误差集中在低权重区域**：占 81% 的实心高斯（α>0.8）`|Δα|` 中位仅 **0.0004~0.0008**，
  误差集中在 α∈[0.05, 0.8] 的半透明 / 稀疏区（对合成贡献本就低）。

| 按参考 α 分桶的 \|Δα\| 中位 | [0.00,0.05) | [0.05,0.20) | [0.20,0.50) | [0.50,0.80) | [0.80,1.01) |
|---|---|---|---|---|---|
| fp16 | 0.0001 | 0.0008 | 0.0017 | 0.0028 | 0.0002 |
| hqq_bs32 | 0.0021 | 0.0138 | 0.0232 | 0.0235 | 0.0004 |

* **量化没有引入新的 WebGPU 缺口**：全部变体覆盖率都是 99.79%，缺失的
  `ConstantOfShape`(8) / `Mod`(2) / `Softplus`(1) 在 fp32 原图里就已存在。
* **CPU 时延不代表目标平台**：4bit 在 CPU 上（约 98 s）比 fp16（约 50 s）还慢 ——
  CPU 没有 4bit 快路径，只多一层 dequantize；`MatMulNBits` 的收益只在 WebGPU / CUDA 上体现。
* 以上均为**单张图**结论，上线前建议换 20~50 张不同类型图像重跑 `04`，确认 p95 不随内容漂移。

## 关键约束

1. **输入分辨率锁死 1536×1536**，与 ONNX 无关：SPN 编码器里 `x2 = x/4` 不再切分、整张作为
   1 个 patch 送入 patch=384 的 ViT ⇒ 必须 `S/4 == 384`。
2. **必须单图导出**：Gaussian decoder 直接消费编码器的 5 张多尺度特征图。
3. **WebGPU 只能用 `MatMulNBits`**：WebGPU EP 没有 `MatMulInteger` / `QLinearConv` /
   `*QuantizeLinear`，ORT 标准 int8 产物会把矩阵乘整体丢回 CPU。
4. **不要用 bf16**：ORT 无 bf16 转换器；与 fp16 同体积但尾数更少（8 < 10 位），
   而本模型最弱的输出正是连续几何量（`opacities` / `singular_values`）。

各步骤的完整踩坑记录（protobuf 2 GB 上限、`Resize.scales`、AWQ/GPTQ 的实测结论等）见
`out/_obsolete/README_full.md` 与 `out/PYTHON_LAYER_FINDINGS.md`。
