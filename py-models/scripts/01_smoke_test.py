"""SHARP 冒烟测试 + 对拍 fixture 导出。

这是纯 PyTorch 侧的基础测试，不涉及 ONNX。用于在导出之前确认：
  1. 依赖可导入，torch 能用目标设备
  2. 官方 checkpoint 能 strict 加载
  3. 输入构造与 sharp/cli/predict.py 完全一致（1536 分辨率、disparity_factor）
  4. 前向能跑通，输出形状/量级符合预期
  5. 内部 monodepth 分支的 I/O 规格
  6. 导出 JS 侧数值对拍所需的 fixtures（out/fixtures/）

第 6 步产出的 fixtures 供 `npx tsx scripts/sharp-compare-fixtures.ts` 使用，
是"JS 实现是否忠实复刻 PyTorch"的唯一判据来源。原先这一步在独立的
05_export_fixtures.py 里，现合并到本脚本，理由：
  * 两者共用同一套加载/构造逻辑（本就需要模型与输入），分文件反而重复；
  * fixtures 必须在**同一次**前向里产生，否则无法保证与上面 4 的数值同源。

用法:
    python scripts/01_smoke_test.py [设备]        # 设备: default/mps/cpu/cuda
    python scripts/01_smoke_test.py cpu --no-export-fixtures
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import (INTERNAL_RESOLUTION, OUT_NAMES, OUT_DIR, load_input, pick_device,
                    build_predictor)

# Gaussians3D 的字段（disparity 不在其中 —— 它由内部 monodepth 分支单独产出）
GAUSS_FIELDS = [n for n in OUT_NAMES if n != "disparity"]

FIXTURE_DIR = OUT_DIR / "fixtures"

FAILED: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILED.append(msg)


# --------------------------------------------------------------------------- #
# 阶段 6：fixtures 导出
# --------------------------------------------------------------------------- #
def export_fixtures(img, disp, ndc, disparity) -> None:
    """导出 JS 侧对拍所需的全部 fixture。

    产出（out/fixtures/）:
        input.npz       image[3,1536,1536] fp32 + 原图尺寸 + f_px + disparity_factor
        ndc.npz         模型输出的 NDC 空间 5 个张量（展平为 [N,C]）+ disparity
        metric.npz      unproject_gaussians 之后的度量空间张量 + intrinsics_resized
        resize.npz      合成图 resize 用例（隔离 JPEG 解码器差异）
        reference.ply   官方 save_ply 的产物，用于 PLY 逐字节对拍

    注意：NDC 基准刻意用 **PyTorch** 输出，而不是 ONNX 输出。
    ONNX 相对 PyTorch 的数值偏差是另一条独立链路（由 04_evaluate.py 负责）。
    fixtures 要回答的是"JS 是否忠实复刻 PyTorch 语义"，基准用 PyTorch 才干净。
    """
    import torch.nn.functional as F
    from sharp.utils import gaussians as gs_utils
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # img/disp/ndc/disparity 由主流程传入，避免重复跑代价高昂的前向（cpu 约 70s/次）

    # ── 原图元信息（JS 侧要据此算 f_px / 反投影）──
    _, _, f_px = sharp_io.load_rgb(
        Path(ROOT, "clones", "ml-depth-pro", "data", "example.jpg")
    )
    orig_w, orig_h = _orig_size()

    # ── input.npz ──
    np.savez(
        FIXTURE_DIR / "input.npz",
        image=img[0].cpu().numpy(),
        orig_width=np.array([orig_w], dtype=np.int32),
        orig_height=np.array([orig_h], dtype=np.int32),
        f_px=np.array([f_px], dtype=np.float32),
        disparity_factor=disp.cpu().numpy().astype(np.float32),
    )
    print(f"      input.npz      原图 {orig_w}x{orig_h} f_px={f_px:.4f}")

    # ── ndc.npz：模型输出（NDC 空间）──
    #
    # 注意 `disparity` 不属 Gaussians3D，它由内部 monodepth 分支单独产出。
    # 但 ONNX 图把它一起导出了（见 02_export_fp16.py 的输出列表），
    # 所以 fixture 也必须存它，否则 JS 侧无从校准这条支路。
    np.savez(
        FIXTURE_DIR / "ndc.npz",
        **{n: getattr(ndc, n)[0].cpu().numpy() for n in GAUSS_FIELDS},
        disparity=disparity[0].cpu().numpy(),
    )
    print(
        f"      ndc.npz        {len(GAUSS_FIELDS)} 个张量 + disparity，"
        f"N={ndc.mean_vectors.shape[1]}"
    )

    # ── intrinsics：严格复刻 `predict.py: predict_image` ──
    #
    # 关键：官方用的不是"裸对角阵"，而是**完整针孔矩阵**（带主点 W/2, H/2），
    # 然后才按行缩放。写成裸对角会让 inv(ndc@K) 多出虚假的 off-diagonal 项
    # （z 列出现非零值），反投影结果完全错。
    #
    # 验证：完整针孔形式下，ndc 矩阵的平移项恰好把主点消掉，
    # inv(ndc@K) 退化为纯对角 (W/(2f), H/(2f), 1)。
    intrinsics = torch.tensor(
        [
            [f_px, 0, orig_w / 2, 0],
            [0, f_px, orig_h / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=ndc.mean_vectors.device,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_RESOLUTION / orig_w
    intrinsics_resized[1] *= INTERNAL_RESOLUTION / orig_h
    intrinsics = intrinsics_resized
    extrinsics = torch.eye(4, dtype=torch.float32, device=ndc.mean_vectors.device)

    unproj = gs_utils.get_unprojection_matrix(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_shape=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
    )

    with torch.no_grad():
        metric = gs_utils.unproject_gaussians(
            gaussians_ndc=ndc,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_shape=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
        )

    np.savez(
        FIXTURE_DIR / "metric.npz",
        mean_vectors=metric.mean_vectors[0].cpu().numpy(),
        singular_values=metric.singular_values[0].cpu().numpy(),
        quaternions=metric.quaternions[0].cpu().numpy(),
        colors=metric.colors[0].cpu().numpy(),
        opacities=metric.opacities[0].cpu().numpy(),
        intrinsics_resized=intrinsics.cpu().numpy(),
        unprojection_matrix=unproj.cpu().numpy(),
    )
    print("      metric.npz     含 intrinsics_resized 与 unprojection_matrix")

    # ── reference.ply：官方 save_ply ──
    save_ply(metric, f_px, (orig_w, orig_h), FIXTURE_DIR / "reference.ply")
    ply_bytes = (FIXTURE_DIR / "reference.ply").stat().st_size
    print(f"      reference.ply  {ply_bytes/1e6:.1f} MB")

    # ── resize.npz：隔离 JPEG 解码器的 resize 用例 ──
    #
    # 整图对拍会同时引入 (a) JPEG 解码差异 与 (b) resize 实现差异，无法区分。
    # 这里用**确定性合成图**当源，JS 与 torch 都从同一组像素出发，即可单独验证 resize。
    #
    # 用**平滑**的合成图（低频正弦叠加）而不是白噪声：
    # 白噪声相邻像素可差 1.0，是物理上不存在的最坏情形，会把采样坐标上
    # 1e-4 的浮点差放大成 ~2e-4 的像素误差，掩盖真正的算法错误。
    # 真实图像（包括本用例的 example.jpg）相邻像素高度相关，误差小几个量级。
    gh, gw = 2268, 3024
    yy, xx = np.meshgrid(
        np.linspace(0, 1, gh, dtype=np.float32),
        np.linspace(0, 1, gw, dtype=np.float32),
        indexing="ij",
    )
    src = np.stack(
        [
            np.sin(3.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy),
            np.sin(1.5 * np.pi * (xx + yy)),
            xx * 0.5 + yy * 0.5,
        ]
    ).astype(np.float32)
    # 加一点确定性高频纹理（周期 ~4px），但仍远不如白噪声剧烈
    src += (np.sin(97.0 * xx) * np.sin(89.0 * yy)).astype(np.float32) * 0.05
    src = src.astype(np.float32)
    src_t = torch.from_numpy(src)[None]
    dst = F.interpolate(
        src_t,
        size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
        mode="bilinear",
        align_corners=True,
    )
    np.savez(
        FIXTURE_DIR / "resize.npz",
        src=src,
        dst=dst[0].numpy(),
        src_size=np.array([gw, gh], dtype=np.int32),
        dst_size=np.array([INTERNAL_RESOLUTION, INTERNAL_RESOLUTION], dtype=np.int32),
        align_corners=np.array([1], dtype=np.int32),
    )
    print(f"      resize.npz     {gw}x{gh} -> {INTERNAL_RESOLUTION}²，平滑合成图，确定性")


def _orig_size() -> tuple[int, int]:
    """原图 (width, height)。"""
    from PIL import Image

    p = Path(ROOT, "clones", "ml-depth-pro", "data", "example.jpg")
    with Image.open(p) as im:
        return im.size


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_fixtures = "--no-export-fixtures" not in sys.argv

    device = pick_device(argv[0] if argv else "default")
    print(f"[env] torch={torch.__version__} numpy={np.__version__} device={device}")

    print("\n[1/6] 构建模型")
    model = build_predictor(device)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"      参数量 = {n_all/1e6:.2f}M  (fp32 {n_all*4/1e9:.2f}GB)")
    check(n_all > 0, "模型有参数")
    check(model.internal_resolution() == INTERNAL_RESOLUTION,
          f"internal_resolution == {INTERNAL_RESOLUTION}")

    print("\n[2/6] 加载权重（已在 build_predictor 内 strict 加载）")
    print(f"      ckpt strict load OK, output_resolution = {model.output_resolution}")

    print("\n[3/6] 构造输入")
    img, disp = load_input()
    img = img.to(device)
    disp = disp.to(device)
    print(f"      image {tuple(img.shape)} {img.dtype} range=[{img.min():.3f}, {img.max():.3f}]")
    print(f"      disparity_factor = {disp.item():.6f}   (= f_px / width)")
    check(tuple(img.shape) == (1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
          f"输入图像为 1x3x{INTERNAL_RESOLUTION}x{INTERNAL_RESOLUTION}")
    check(0.0 <= float(img.min()) and float(img.max()) <= 1.0, "图像值域在 [0,1]")

    print("\n[4/6] 前向推理")
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

    print("\n[5/6] 内部 monodepth 分支")
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

    if do_fixtures:
        print(f"\n[6/6] 导出对拍 fixtures -> {FIXTURE_DIR.relative_to(ROOT)}")
        export_fixtures(img, disp, out, md.disparity)
        check((FIXTURE_DIR / "input.npz").exists(), "input.npz 已生成")
        check((FIXTURE_DIR / "ndc.npz").exists(), "ndc.npz 已生成")
        check((FIXTURE_DIR / "metric.npz").exists(), "metric.npz 已生成")
        check((FIXTURE_DIR / "resize.npz").exists(), "resize.npz 已生成")
        check((FIXTURE_DIR / "reference.ply").exists(), "reference.ply 已生成")
    else:
        print("\n[6/6] 跳过 fixtures 导出（--no-export-fixtures）")

    print()
    if FAILED:
        print(f"[smoke] 失败 {len(FAILED)} 项:")
        for f in FAILED:
            print(f"   - {f}")
        sys.exit(1)
    print("[smoke] 全部通过")


if __name__ == "__main__":
    main()
