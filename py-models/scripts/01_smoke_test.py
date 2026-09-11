"""SHARP 冒烟测试：环境 / 权重 / 前向 / 输出规格。

这是纯 PyTorch 侧的基础测试，不涉及 ONNX。用于在导出之前确认：
  1. 依赖可导入，torch 能用目标设备
  2. 官方 checkpoint 能 strict 加载
  3. 输入构造与 sharp/cli/predict.py 完全一致（1536 分辨率、disparity_factor）
  4. 前向能跑通，输出形状/量级符合预期
  5. 内部 monodepth 分支的 I/O 规格

用法:
    python scripts/01_smoke_test.py [设备]      # 设备: default/mps/cpu/cuda
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from common import (INTERNAL_RESOLUTION, OUT_NAMES, load_input, pick_device,
                    build_predictor)

# Gaussians3D 的字段（disparity 不在其中 —— 它由内部 monodepth 分支单独产出）
GAUSS_FIELDS = [n for n in OUT_NAMES if n != "disparity"]

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILED.append(msg)


def main() -> None:
    device = pick_device(sys.argv[1] if len(sys.argv) > 1 else "default")
    print(f"[env] torch={torch.__version__} numpy={np.__version__} device={device}")

    print("\n[1/5] 构建模型")
    model = build_predictor(device)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"      参数量 = {n_all/1e6:.2f}M  (fp32 {n_all*4/1e9:.2f}GB)")
    check(n_all > 0, "模型有参数")
    check(model.internal_resolution() == INTERNAL_RESOLUTION,
          f"internal_resolution == {INTERNAL_RESOLUTION}")

    print("\n[2/5] 加载权重（已在 build_predictor 内 strict 加载）")
    print(f"      ckpt strict load OK, output_resolution = {model.output_resolution}")

    print("\n[3/5] 构造输入")
    img, disp = load_input()
    img = img.to(device)
    disp = disp.to(device)
    print(f"      image {tuple(img.shape)} {img.dtype} range=[{img.min():.3f}, {img.max():.3f}]")
    print(f"      disparity_factor = {disp.item():.6f}   (= f_px / width)")
    check(tuple(img.shape) == (1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
          f"输入图像为 1x3x{INTERNAL_RESOLUTION}x{INTERNAL_RESOLUTION}")
    check(0.0 <= float(img.min()) and float(img.max()) <= 1.0, "图像值域在 [0,1]")

    print("\n[4/5] 前向推理")
    t0 = time.time()
    with torch.no_grad():
        out = model(img, disp)
    dt = time.time() - t0
    print(f"      耗时 {dt:.2f}s ({device})")
    for name in GAUSS_FIELDS:
        t = getattr(out, name)
        print(f"      {name:18s} shape={tuple(t.shape)} "
              f"min={t.min().item():.4f} max={t.max().item():.4f}")
    check(all(isinstance(getattr(out, k), torch.Tensor) for k in GAUSS_FIELDS),
          f"Gaussians3D 包含全部 {len(GAUSS_FIELDS)} 个字段")
    check(torch.isfinite(out.mean_vectors).all().item(), "mean_vectors 无 NaN/Inf")
    check(torch.isfinite(out.quaternions).all().item(), "quaternions 无 NaN/Inf")
    check(torch.isfinite(out.colors).all().item(), "colors 无 NaN/Inf")
    check(torch.isfinite(out.opacities).all().item(), "opacities 无 NaN/Inf")
    check(float(out.opacities.min()) >= 0.0 and float(out.opacities.max()) <= 1.0,
          "opacities 值域在 [0,1]")

    print("\n[5/5] 内部 monodepth 分支")
    with torch.no_grad():
        md = model.monodepth_model(img)
    d = md.disparity
    print(f"      disparity {tuple(d.shape)} range=[{d.min():.6f}, {d.max():.6f}]")
    depth = disp[:, None, None, None] / d.clamp(min=1e-4, max=1e4)
    print(f"      度量深度 {tuple(depth.shape)} range=[{depth.min():.4f}, {depth.max():.4f}] m")
    print(f"      encoder_features: {[tuple(f.shape) for f in md.encoder_features]}")
    print(f"      decoder_features: {tuple(md.decoder_features.shape)}")
    check(d.shape[-2:] == (INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
          f"disparity 分辨率为 {INTERNAL_RESOLUTION}x{INTERNAL_RESOLUTION}")
    check(len(md.encoder_features) == 5, "monodepth 编码器输出 5 张多尺度特征图")

    print()
    if FAILED:
        print(f"[smoke] 失败 {len(FAILED)} 项:")
        for f in FAILED:
            print(f"   - {f}")
        sys.exit(1)
    print("[smoke] 全部通过")


if __name__ == "__main__":
    main()
