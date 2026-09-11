"""SHARP -> ONNX 流水线的共享工具。

被四个脚本复用：

    01_smoke_test.py      PyTorch 侧冒烟测试（模型/权重/前向/输出规格）
    02_export_fp16.py     fp32 基座 -> fp16 ONNX
    03_export_mm4f16.py   fp32 基座 -> 4bit MatMul 权重(MatMulNBits) + 其余 fp16
    04_evaluate.py        以上模型与 fp32 参考的对比评估

约定：
    * 所有中间/最终产物都在 ``out/onnx/`` 下，命名见各函数。
    * ``sharp.onnx``        —— torch.onnx.export 原始产物（保留 Identity 包装）
    * ``sharp_clean.onnx``  —— 折叠 Identity 之后（4bit 量化的输入）
    * ``ref_outputs.npz``   —— fp32 ORT 参考张量与输入，评估基准
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx

# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
CLONES = ROOT / "clones"
SHARP_SRC = CLONES / "ml-sharp" / "src"
CKPT = CLONES / "ml-sharp" / "downloads" / "sharp_2572gikvuh.pt"
# 用一张真实的单张照片作为统一测试输入（ml-sharp 的 teaser.jpg 是拼图，不适合）
IMG = CLONES / "ml-depth-pro" / "data" / "example.jpg"

OUT_DIR = ROOT / "out"
ONNX_DIR = OUT_DIR / "onnx"
LOG_DIR = OUT_DIR / "logs"

# SHARP 的 SPN 编码器结构锁死了内部推理分辨率必须是 1536（见 README）
INTERNAL_RESOLUTION = 1536

# 导出模型的输出顺序
OUT_NAMES = [
    "mean_vectors",
    "singular_values",
    "quaternions",
    "colors",
    "opacities",
    "disparity",
]

sys.path.insert(0, str(SHARP_SRC))

# WebGPU EP 有 kernel 的算子（摘自 onnxruntime/js/web/docs/webgpu-operators.md，
# ai.onnx 域）+ com.microsoft 贡献算子中 WebGPU 自己实现的那些。
WEBGPU_OPS = {
    "Abs", "Acos", "Acosh", "Add", "ArgMax", "ArgMin", "Asin", "Asinh", "Atan", "Atanh",
    "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip", "Concat", "Conv",
    "ConvTranspose", "Cos", "Cosh", "CumSum", "DFT", "DepthToSpace", "DequantizeLinear",
    "Div", "Einsum", "Elu", "Equal", "Erf", "Exp", "Expand", "Flatten", "Floor", "Gather",
    "GatherElements", "GatherND", "Gemm", "GlobalAveragePool", "GlobalMaxPool", "Greater",
    "GreaterOrEqual", "GridSample", "HardSigmoid", "HardSwish", "If", "InstanceNormalization",
    "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log", "MatMul", "MaxPool",
    "MemcpyFromHost", "MemcpyToHost", "Mul", "Neg", "Not", "Pad", "Pow", "Range",
    "Reciprocal", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
    "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu",
    "Reshape", "Resize", "ScatterND", "Shape", "Sigmoid", "SimplifiedLayerNormalization",
    "Sin", "Sinh", "Slice", "Softmax", "Split", "Sqrt", "Squeeze", "Sub", "Tan", "Tanh",
    "ThresholdedRelu", "Tile", "Transpose", "Unsqueeze", "Where",
    # com.microsoft：WebGPU EP 有专门的 MatMulNBits WGSL kernel
    "MatMulNBits", "Attention", "MultiHeadAttention", "Gelu", "FastGelu", "BiasAdd",
    "GroupQueryAttention", "RotaryEmbedding", "GatherBlockQuantized", "QuickGelu",
}

# 非计算节点：形状/常量搬运，不参与 EP 覆盖统计
NON_COMPUTE_OPS = {"Constant", "Identity"}


# --------------------------------------------------------------------------- #
# 设备
# --------------------------------------------------------------------------- #
def pick_device(arg: str = "default"):
    import torch

    if arg != "default":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# 模型与输入
# --------------------------------------------------------------------------- #
def build_predictor(device=None):
    """构建 SHARP predictor 并 strict 加载官方权重。"""
    import torch
    from sharp.models import PredictorParams, create_predictor

    model = create_predictor(PredictorParams())
    model.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    if device is not None:
        model.to(device)
    return model


def load_input():
    """复刻 sharp/cli/predict.py 的输入构造：返回 (img1536, disparity_factor)。

    img1536: torch.Tensor [1,3,1536,1536]，值域 [0,1]
    disparity_factor: torch.Tensor [1]，= f_px / width
    """
    import torch
    import torch.nn.functional as F
    from sharp.utils import io as sharp_io

    image, _, f_px = sharp_io.load_rgb(IMG)
    h, w = image.shape[:2]
    img = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    img1536 = F.interpolate(
        img[None], size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
        mode="bilinear", align_corners=True,
    )
    disp = torch.tensor([f_px / w], dtype=torch.float32)
    return img1536, disp


# --------------------------------------------------------------------------- #
# fp32 基座导出
# --------------------------------------------------------------------------- #
def _raw_dir() -> Path:
    return OUT_DIR / "onnx_raw"


def export_fp32_base(opset: int = 17, force: bool = False) -> Path:
    """把 SHARP 全图导出为单个 ONNX（含外部数据）。

    为什么是单图：Gaussian decoder 直接消费 monodepth 编码器的 5 张多尺度特征图，
    拆图需要交换约 878MB(fp32) 的中间张量，工程上无意义。
    """
    out = ONNX_DIR / "sharp.onnx"
    if out.exists() and not force:
        print(f"[base] 复用已有 {out.name}")
        return out

    import torch
    import torch.nn as nn

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = _raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / "sharp.onnx"

    from sharp.models import PredictorParams, create_predictor  # noqa: F401

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image, disparity_factor):
            md = self.model.monodepth_model(image)
            g = self.model(image, disparity_factor)
            return (g.mean_vectors, g.singular_values, g.quaternions,
                    g.colors, g.opacities, md.disparity)

    model = build_predictor()
    wrapper = Wrapper(model).eval()
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"[base] params={n/1e6:.2f}M -> fp32 {n*4/1e9:.2f}GB / fp16 {n*2/1e9:.2f}GB")

    img1536, disp = load_input()
    print(f"[base] torch.onnx.export opset={opset} ...")
    import time

    t0 = time.time()
    torch.onnx.export(
        wrapper, (img1536, disp), str(raw),
        input_names=["image", "disparity_factor"], output_names=OUT_NAMES,
        opset_version=opset, dynamo=False, external_data=True,
        do_constant_folding=True, optimize=False,
    )
    print(f"[base] 导出耗时 {time.time()-t0:.1f}s")

    # torch.onnx.export 会为每个 initializer 写一个散装文件，合并成单个 .data
    m = onnx.load(str(raw), load_external_data=True)
    for init in m.graph.initializer:
        init.ClearField("external_data")
        init.data_location = onnx.TensorProto.DEFAULT
    save_external(m, out, location="sharp.onnx.data")
    del m
    print(f"[base] -> {out.name} {model_size(out)/1e9:.3f} GB")
    return out


def fold_identity(src: Path | None = None, dst: Path | None = None) -> Path:
    """把 ``Identity(initializer)`` 折叠掉，得到 4bit 量化的输入图。

    为什么必须做：torch.onnx.export 把 nn.Linear 的权重导出成
    ``MatMul(x, Identity(W))``。ORT 的 weight-only 4bit 量化靠
    "B 是常量 initializer" 来识别目标节点，Identity 的输出不被认作常量，
    于是这批权重完全不会被量化、却仍以 fp32 留在图里（实测 190 个，约 2.4GB）。
    """
    src = src or (ONNX_DIR / "sharp.onnx")
    dst = dst or (ONNX_DIR / "sharp_clean.onnx")
    if dst.exists():
        print(f"[fold] 复用已有 {dst.name}")
        return dst

    print("[fold] 折叠 Identity-of-initializer ...")
    m = onnx.load(str(src), load_external_data=True)
    init_names = {i.name for i in m.graph.initializer}

    alias: dict[str, str] = {}
    keep = []
    for node in m.graph.node:
        if (node.op_type == "Identity" and len(node.input) == 1
                and node.input[0] in init_names and node.input[0] not in alias):
            alias[node.output[0]] = node.input[0]
        else:
            keep.append(node)

    if not alias:
        print("[fold] 没有可折叠的 Identity")
        del m
        return src

    del m.graph.node[:]
    m.graph.node.extend(keep)

    def resolve(n: str) -> str:
        seen = set()
        while n in alias and n not in seen:
            seen.add(n)
            n = alias[n]
        return n

    for node in m.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name)

    used = {i for n in m.graph.node for i in n.input if i}
    inits = [i for i in m.graph.initializer if i.name in used]
    dropped = len(m.graph.initializer) - len(inits)
    del m.graph.initializer[:]
    m.graph.initializer.extend(inits)
    print(f"[fold] 折叠 {len(alias)} 个 Identity，移除 {dropped} 个失引 initializer")

    save_external(m, dst, location=dst.name + ".data")
    del m
    print(f"[fold] -> {dst.name} {model_size(dst)/1e9:.3f} GB")
    return dst


def ensure_ref_outputs(force: bool = False) -> Path:
    """跑一次 fp32 ONNX，保存参考张量与输入，作为所有评估的基准。"""
    ref_path = ONNX_DIR / "ref_outputs.npz"
    if ref_path.exists() and not force:
        print(f"[ref] 复用已有 {ref_path.name}")
        return ref_path

    import onnxruntime as ort

    base = export_fp32_base()
    img1536, disp = load_input()
    feeds = {"image": img1536.numpy(), "disparity_factor": disp.numpy()}

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(base), so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    np.savez_compressed(
        ref_path, **{k: v for k, v in zip(OUT_NAMES, outs)}, **feeds
    )
    print(f"[ref] 参考张量 -> {ref_path.name}")
    return ref_path


# --------------------------------------------------------------------------- #
# ONNX 存取与 fp16 转换
# --------------------------------------------------------------------------- #
def save_external(model: onnx.ModelProto, path: Path, location: str | None = None) -> None:
    """以「单文件外部数据」方式保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(path),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=location or (path.name + ".data"),
        size_threshold=1024, convert_attribute=False,
    )


def model_size(path: Path) -> int:
    """模型文件 + 外部数据文件的总字节数。"""
    total = path.stat().st_size
    data = path.with_name(path.name + ".data")
    if data.exists():
        total += data.stat().st_size
    return total


def fix_resize_scales(model: onnx.ModelProto) -> int:
    """把 Resize 的 scales 常量还原成 float32。

    ONNX 规范要求 Resize 的 scales 是 tensor(float)。float16 转换器会把 Constant
    节点的 value 属性一并转成 fp16，ORT 加载时报
    "Type 'tensor<float16>' of input parameter ... of operator (Resize) is invalid"。
    """
    producer = {o: n for n in model.graph.node for o in n.output}
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Resize" or len(node.input) < 3:
            continue
        src = producer.get(node.input[2])
        if src is None or src.op_type != "Constant":
            continue
        for attr in src.attribute:
            if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT16:
                arr = onnx.numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(onnx.numpy_helper.from_array(arr, attr.t.name))
                fixed += 1
    return fixed


def convert_to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """整图 fp16 转换（含 Resize.scales 修复）。

    两个必须知道的坑：
      * ``disable_shape_infer=True`` 是必需的 —— 模型权重 2.6GB，onnx.shape_inference
        会先 SerializeToString()，直接撞上 protobuf 的 2GB 上限（EncodeError）。
      * **不要**把 Resize 放进 ``op_block_list``：被 block 的 op 自己保持 fp32，
        但它的输入仍会被转成 fp16，反而触发 Resize 的类型校验失败。
        Resize.scales 用 ``fix_resize_scales`` 事后修。

    还有一个坑在**调用方**：如果入参 model 是用 ``onnx.load(..., load_external_data=False)``
    打开的，先用 ``materialize_external(model, model_dir)`` 处理，否则会报
    ``ValidationError: Data of TensorProto (...) should be stored in <x>.data,
    but it is not regular file``（详见该函数的说明）。
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m16 = convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
        op_block_list=["RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike"],
    )
    n = fix_resize_scales(m16)
    print(f"[fp16] Resize.scales 还原为 float32: {n} 处")
    return m16


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def pearson(a: np.ndarray, b: np.ndarray, stride: int = 37) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a.ravel()[::stride], b.ravel()[::stride])[0, 1])


def gaussian_metrics(outs: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict:
    """3DGS 语义指标：把张量误差换算成"看上去差多少"。

    PSNR 只说明整体能量误差，对 3DGS 有误导性 —— 绝大多数高斯是近乎透明的
    （alpha 接近 0），它们的误差对最终画面没有贡献。所以这里按**参考 alpha 分桶**，
    并给出 alpha 加权的颜色误差。
    """
    buckets = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]

    a_ref = ref["opacities"].astype(np.float32).reshape(-1)
    a = outs["opacities"].astype(np.float32).reshape(-1)
    c_ref = ref["colors"].astype(np.float32).reshape(-1, 3)
    c = outs["colors"].astype(np.float32).reshape(-1, 3)
    da = np.abs(a - a_ref)
    dc = np.abs(c - c_ref)

    out = {"buckets": [], "n_gaussians": int(a_ref.size)}
    for lo, hi in buckets:
        m = (a_ref >= lo) & (a_ref < hi)
        if not m.any():
            continue
        out["buckets"].append({
            "range": [lo, hi],
            "share_pct": float(m.mean() * 100),
            "dalpha_p50": float(np.median(da[m])),
        })

    vis, faint = a_ref > 0.5, a_ref < 0.1
    out["visible_gt0.5"] = {
        "share_pct": float(vis.mean() * 100),
        "dalpha_p50": float(np.median(da[vis])),
        "dalpha_p95": float(np.percentile(da[vis], 95)),
    }
    out["faint_lt0.1"] = {
        "share_pct": float(faint.mean() * 100),
        "dalpha_p50": float(np.median(da[faint])),
        "dalpha_p95": float(np.percentile(da[faint], 95)),
    }
    # alpha 加权的颜色误差 ≈ 实际看到的像素误差，换算到 8bit 级
    w = a_ref / max(a_ref.sum(), 1e-9)
    out["color_8bit_alpha_weighted"] = float((w[:, None] * dc).sum() * 255)
    out["color_8bit_median"] = float(np.median(dc) * 255)
    out["alpha_mass_rel_dev_pct"] = float(abs(a.sum() - a_ref.sum()) / a_ref.sum() * 100)

    # 高斯中心位移，以场景量程为尺度
    mv = np.linalg.norm(outs["mean_vectors"] - ref["mean_vectors"], axis=-1)
    scene = float(np.abs(ref["mean_vectors"]).max())
    out["center_shift_p50_pct"] = float(np.median(mv) / scene * 100)
    out["center_shift_p95_pct"] = float(np.percentile(mv, 95) / scene * 100)
    return out


def webgpu_coverage(path: Path) -> dict:
    """计算节点（剔除 Constant/Identity）里 WebGPU 有 kernel 的比例。"""
    from collections import Counter

    m = onnx.load(str(path), load_external_data=False)
    ops = Counter(n.op_type for n in m.graph.node)
    comp = {k: v for k, v in ops.items() if k not in NON_COMPUTE_OPS}
    total = sum(comp.values())
    missing = {k: v for k, v in comp.items()
               if k not in WEBGPU_OPS and not k.startswith("com.")}
    return {
        "compute_nodes": total,
        "coverage_pct": round((total - sum(missing.values())) / total * 100, 2) if total else 0.0,
        "missing": missing,
    }


def run_onnx(path: Path, feeds: dict, providers=("CPUExecutionProvider",)):
    """加载并推理，返回 (输出列表, 用时秒)。输入 dtype 按模型签名自动匹配。"""
    import time

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=list(providers))
    feeds = dict(feeds)
    for inp in sess.get_inputs():
        if inp.type == "tensor(float16)":
            feeds[inp.name] = feeds[inp.name].astype(np.float16)
    t0 = time.time()
    outs = sess.run(None, feeds)
    return outs, time.time() - t0
