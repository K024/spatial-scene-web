"""导出 mm4f16 ONNX 模型：MatMul 权重 4bit（MatMulNBits）+ 其余部分 fp16。

为什么是这条路线（而非 int8）：WebGPU EP 的算子表里有 ``MatMulNBits``（并且有
专门的 WGSL kernel，含 DP4A / subgroup-matrix 的 int8 路径），却没有
``MatMulInteger`` / ``QLinearConv`` / ``DynamicQuantizeLinear`` / ``QuantizeLinear``。
ORT 的标准 int8 产物在 WebGPU 上会把全部矩阵乘和卷积丢回 CPU —— 正是社区反复
报告的"int8 比 fp16 还慢"的根因。

参数默认值（均为固定约定，不做其它取值）：

    bits            = 4
    block_size      = 32 与 128（两者都导出，用于对比）
    is_symmetric    = False
    accuracy_level  = 2   （MatMulNBits 内部把输入 A 降为 fp16 计算）

算法：

    rtn   —— ORT 自带的 block-wise round-to-nearest（data-free）
    hqq   —— ORT 自带的 HQQ（half-quadratic quantization，data-free）

产物命名：``sharp_mm4f16_{algo}_bs{block_size}.onnx``

用法:
    python scripts/03_export_mm4f16.py [rtn|hqq|all]

--------------------------------------------------------------------------------
关于 activation-aware 4bit（AWQ / GPTQ）：已试过，未采用
--------------------------------------------------------------------------------
AWQ 与 GPTQ 都实现并实测过，结论是不值得：本模型上 RTN/HQQ 已足够好且不需要校准数据。
细节与代码见 ``out/_obsolete/README_full.md``、``out/_obsolete/awq/05_export_awq.py``。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import onnx

from common import (ONNX_DIR, convert_to_fp16, ensure_ref_outputs, fold_identity,
                    model_size, save_external)

BITS = 4
BLOCK_SIZES = (32, 128)
ALGOS = ("rtn", "hqq")
IS_SYMMETRIC = False
ACCURACY_LEVEL = 2


def out_name(algo: str, block_size: int) -> str:
    return f"sharp_mm4f16_{algo}_bs{block_size}.onnx"


def quantize_ort(folded: Path, algo: str, block_size: int) -> onnx.ModelProto:
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig, HQQWeightOnlyQuantConfig, MatMulNBitsQuantizer)

    if algo == "rtn":
        cfg = DefaultWeightOnlyQuantConfig(
            block_size=block_size, is_symmetric=IS_SYMMETRIC,
            accuracy_level=ACCURACY_LEVEL, bits=BITS)
    elif algo == "hqq":
        # HQQ 不支持 symmetric（其 scheme 由算法自身决定，恒为 asym），
        # 也不接受 accuracy_level —— 后者在下面用 set_accuracy_level 补上。
        cfg = HQQWeightOnlyQuantConfig(block_size=block_size, bits=BITS)
    else:
        raise ValueError(f"未知算法 {algo}，本脚本只支持 {ALGOS}")

    t0 = time.time()
    q = MatMulNBitsQuantizer(
        str(folded), bits=BITS, block_size=block_size,
        is_symmetric=IS_SYMMETRIC, accuracy_level=ACCURACY_LEVEL,
        algo_config=cfg, op_types_to_quantize=("MatMul",),
    )
    q.process()
    proto = getattr(q.model, "model", q.model)
    if callable(proto):
        proto = proto()
    print(f"[{algo} bs={block_size}] ORT 量化完成 {time.time()-t0:.1f}s")
    return proto


def set_accuracy_level(model: onnx.ModelProto, level: int = ACCURACY_LEVEL) -> int:
    """给缺 accuracy_level 属性的 MatMulNBits 节点补上（HQQ 路径不会写这个属性）。"""
    n = 0
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        if any(a.name == "accuracy_level" for a in node.attribute):
            continue
        node.attribute.append(onnx.helper.make_attribute("accuracy_level", level))
        n += 1
    return n


def finalize(proto: onnx.ModelProto, out: Path) -> None:
    m16 = convert_to_fp16(proto)
    for f in list(ONNX_DIR.glob(out.name + "*")):
        f.unlink()
    save_external(m16, out, location=out.name + ".data")
    del m16


def summarize(out: Path) -> None:
    from collections import Counter

    m = onnx.load(str(out), load_external_data=False)
    ops = Counter(n.op_type for n in m.graph.node)
    nb = [n for n in m.graph.node if n.op_type == "MatMulNBits"]
    attrs = {}
    for a in ("bits", "block_size", "accuracy_level"):
        vals = {next((x.i for x in n.attribute if x.name == a), None) for n in nb}
        attrs[a] = sorted(v for v in vals if v is not None) or ["(未设置)"]
    n_zp = sum(1 for n in nb if len(n.input) >= 4)
    print(f"      MatMulNBits {len(nb)} 个；属性 {attrs}；带 zero_point 的 {n_zp} 个")
    print(f"      MatMul(残留) {ops['MatMul']} 个（注意力 QK^T/attn·V，无常数权重）")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = list(ALGOS) if which == "all" else [which]
    for a in targets:
        if a not in ALGOS:
            raise SystemExit(f"未知算法 {a}，可选 {ALGOS}")

    folded = fold_identity()
    ensure_ref_outputs()

    for algo in targets:
        for bs in BLOCK_SIZES:
            out = ONNX_DIR / out_name(algo, bs)
            print(f"\n{'='*72}\n[mm4f16] {algo}  block_size={bs}  "
                  f"is_symmetric={IS_SYMMETRIC}  accuracy_level={ACCURACY_LEVEL}\n{'='*72}")
            try:
                proto = quantize_ort(folded, algo, bs)
                if algo == "hqq":
                    n = set_accuracy_level(proto)
                    print(f"[hqq bs={bs}] 补写 accuracy_level={ACCURACY_LEVEL}: {n} 个节点")
                finalize(proto, out)
                del proto
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"[mm4f16] {algo} bs={bs} 失败: {type(e).__name__}: {e}")
                continue
            print(f"[mm4f16] -> {out.name}  {model_size(out)/1e9:.3f} GB")
            summarize(out)

    print("\n[mm4f16] 下一步: python scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
