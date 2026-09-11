"""评估 fp16 与全部 mm4f16 变体（统一口径，基准 = fp32 ORT 输出）。

给出四类信息：
  1. 体积与压缩比
  2. 逐输出 PSNR / pearson_r
  3. 3DGS 语义指标 —— PSNR 对 3DGS 有误导性：绝大多数高斯是近乎透明的
     （alpha≈0），它们的误差对最终画面没有贡献。所以按**参考 alpha 分桶**统计
     |Δα|，并给出 alpha 加权的颜色误差（≈实际看到的像素误差）。
  4. WebGPU EP 计算节点覆盖率与缺失算子。

用法:
    python scripts/04_evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from common import (ONNX_DIR, OUT_NAMES, gaussian_metrics, model_size, pearson,
                    psnr, run_onnx, webgpu_coverage)

REF = ONNX_DIR / "ref_outputs.npz"
FP16 = ONNX_DIR / "sharp_fp16.onnx"

BUCKETS = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]


def discover(only: str | None = None) -> list[tuple[str, Path]]:
    """按固定顺序列出待评估模型：fp32 基座 / fp16 / mm4f16 各变体。

    传入 ``only`` 时只保留 tag 含该子串的变体（例如 ``hqq_bs32``）。
    """
    items: list[tuple[str, Path]] = []
    base = ONNX_DIR / "sharp.onnx"
    if base.exists():
        items.append(("fp32", base))
    if FP16.exists():
        items.append(("fp16", FP16))
    for p in sorted(ONNX_DIR.glob("sharp_mm4f16_*.onnx")):
        # sharp_mm4f16_rtn_bs32.onnx -> rtn_bs32
        items.append((p.stem.replace("sharp_mm4f16_", ""), p))
    if only:
        items = [(t, p) for t, p in items if only in t]
    return items


def evaluate(tag: str, path: Path, ref: dict) -> dict:
    size = model_size(path)
    print(f"\n{'='*80}\n[{tag}]  {path.name}   体积 {size/1e9:.3f} GB\n{'='*80}")

    feeds = {"image": ref["image"], "disparity_factor": ref["disparity_factor"]}
    outs, dt = run_onnx(path, feeds)
    print(f"  CPU 推理 {dt:.1f}s")

    res: dict = {"tag": tag, "file": path.name, "size_gb": size / 1e9, "latency_s": dt}
    o = dict(zip(OUT_NAMES, outs))

    if tag == "fp32":
        print("  （基准自身，误差为 0）")
        res["per_output"] = {k: {"psnr_db": 99.0, "pearson_r": 1.0} for k in OUT_NAMES}
    else:
        print(f"  {'输出':<17}{'PSNR(dB)':>10}{'pearson_r':>11}")
        res["per_output"] = {}
        for k in OUT_NAMES:
            r = np.asarray(ref[k], dtype=np.float32)
            a = np.asarray(o[k], dtype=np.float32)
            p = psnr(a, r, float(np.abs(r).max()))
            c = pearson(a, r)
            res["per_output"][k] = {"psnr_db": p, "pearson_r": c}
            print(f"  {k:<17}{p:>10.2f}{c:>11.5f}")

    cov = webgpu_coverage(path)
    res["webgpu"] = cov
    print(f"  [WebGPU] 计算节点 {cov['compute_nodes']}，覆盖 {cov['coverage_pct']:.2f}%"
          f"  未实现: {cov['missing'] or '无'}")

    if tag != "fp32":
        g = gaussian_metrics({k: np.asarray(o[k], dtype=np.float32) for k in OUT_NAMES}, ref)
        res["gaussians"] = g
        vis = g["visible_gt0.5"]
        print(f"  [3DGS] 可见(α>0.5)占 {vis['share_pct']:.1f}%  "
              f"|Δα| 中位 {vis['dalpha_p50']:.4f} / p95 {vis['dalpha_p95']:.4f}")
        print(f"         可见颜色误差(alpha 加权) {g['color_8bit_alpha_weighted']:.3f}/255  "
              f"总 alpha 质量偏差 {g['alpha_mass_rel_dev_pct']:.3f}%  "
              f"中心位移中位 {g['center_shift_p50_pct']:.3f}% 场景量程")
    return res


def main() -> None:
    if not REF.exists():
        raise SystemExit("缺少 out/onnx/ref_outputs.npz —— 请先运行 02 或 03 脚本")
    ref = dict(np.load(REF))

    only = sys.argv[1] if len(sys.argv) > 1 else None
    items = discover(only)
    if not items:
        hint = f"（过滤条件 {only!r}）" if only else ""
        raise SystemExit(f"out/onnx/ 下没有可评估的模型{hint} —— 请先运行 02 / 03 / 05 脚本")
    print(f"待评估: {[t for t, _ in items]}")

    report = [evaluate(tag, path, ref) for tag, path in items]
    base_size = None
    if not only:                       # 只有全量评估时才有 fp32 基准可算压缩比
        base_size = next((r["size_gb"] for r in report if r["tag"] == "fp32"), None)
    print(f"\n\n{'='*100}\n总表\n{'='*100}")
    hdr = (f"{'变体':<12}{'体积GB':>9}{'压缩':>7}" +
           "".join(f"{k[:9]:>11}" for k in ("disparity", "opacities", "colors")) +
           f"{'可见色8bit':>12}{'实心α>0.8':>11}{'WebGPU':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in report:
        po = r["per_output"]
        comp = f"{base_size/r['size_gb']:.2f}x" if base_size else "-"
        solid = next((b["dalpha_p50"] for b in r.get("gaussians", {}).get("buckets", [])
                      if b["range"][0] == 0.8), float("nan"))
        color = r.get("gaussians", {}).get("color_8bit_alpha_weighted", float("nan"))
        print(f"{r['tag']:<12}{r['size_gb']:9.3f}{comp:>7}" +
              "".join(f"{po[k]['psnr_db']:11.2f}" for k in ("disparity", "opacities", "colors")) +
              f"{color:12.3f}{solid:11.4f}{r['webgpu']['coverage_pct']:8.2f}%")

    # ---------------- α 分桶 ----------------
    print(f"\n{'='*100}\n按参考 alpha 分桶的 |Δα| 中位（误差集中在哪）\n{'='*100}")
    hdr = f"{'变体':<12}" + "".join(f"[{lo:.2f},{hi:.2f})".rjust(16) for lo, hi in BUCKETS)
    print(hdr)
    print("-" * len(hdr))
    for r in report:
        if "gaussians" not in r:
            continue
        row = f"{r['tag']:<12}"
        for lo, hi in BUCKETS:
            b = next((x for x in r["gaussians"]["buckets"] if x["range"] == [lo, hi]), None)
            row += f"{b['dalpha_p50']:16.4f}" if b else f"{'-':>16}"
        print(row)

    # ---------------- 缺失算子 ----------------
    print("\nWebGPU 缺失算子:")
    for r in report:
        print(f"  {r['tag']:<12} {r['webgpu']['missing'] or '无'}")

    out = ONNX_DIR / (f"eval_report_{only}.json" if only else "eval_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
