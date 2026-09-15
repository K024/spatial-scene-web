"""导出 fp16 ONNX 模型。

用途：**可移植目标**。fp16 在 CPU / CUDA / DirectML / CoreML / WebGPU 上都有
kernel 实现，是唯一的"全 EP 无缺口"格式。

为什么不是 bf16：
  * ORT 只提供 convert_float_to_float16，没有 bf16 转换器；
  * bf16 与 fp16 同为 2 字节/参数，体积收益为零；
  * bf16 尾数 8 位 < fp16 的 10 位，而本模型最弱的输出恰是连续几何量
    （opacities / singular_values），精度只会更差；
  * CoreML 的计算类型是 fp16/fp32，bf16 不属于其中。

用法:
    python scripts/02_export_fp16.py
"""
from __future__ import annotations

import onnx

from common import (ONNX_DIR, convert_to_fp16, ensure_ref_outputs,
                    export_fp32_base, model_size, save_external)

OUT = ONNX_DIR / "sharp_fp16.onnx"


def main() -> None:
    # fp32 基座（保留 Identity 包装的原始导出图）
    base = export_fp32_base()
    # 评估基准：fp32 ORT 输出
    ensure_ref_outputs()

    print(f"\n[fp16] 载入 {base.name} ...")
    m = onnx.load(str(base), load_external_data=True)
    m16 = convert_to_fp16(m)
    del m

    for f in list(ONNX_DIR.glob(OUT.name + "*")):
        f.unlink()
    save_external(m16, OUT, location="sharp_fp16.onnx.data")
    del m16

    print(f"[fp16] -> {OUT.name}  {model_size(OUT)/1e9:.3f} GB")
    print("[fp16] 下一步: python scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
