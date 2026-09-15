/**
 * PLY 颜色字段 -> 渲染用线性 RGB。
 *
 * `save_ply` 写出的 `f_dc_*` 是**已经做过 linearRGB->sRGB 的** degree-0
 * 球谐系数（见 `gaussians.py: save_ply` 的注释：为了公开渲染器不做 gamma
 * 也能看）。所以读回来的链路是：
 *
 *     线性 RGB --(SH)--> f_dc --(读回)--> sRGB --(sRGB->linear)--> 线性 RGB
 *
 * 后两步就是本文件。**为什么非要转回线性**：公开渲染器直接在 sRGB 空间混合
 * （把结果原样输出），而 SHARP 自己的渲染器是「线性混合 + 最后统一转 sRGB」。
 * 两者在单层（alpha=1）时完全一致，但在半透明重叠处线性混合才物理正确
 * （sRGB 空间混合会让边缘偏暗）。本项目渲染到 RGBA16F 再做线性->sRGB 输出，
 * 因此这里必须回到线性。
 *
 * 这里把「SH 解码 + sRGB 阈值分段」融合成一个循环，避免为 1.18M 高斯
 * 额外分配两个 14 MB 中间数组；数学式子与 `sharp/colorspace.ts`
 * （`sRGB2linearRGB`）和 `sharp/linalg.ts`（`sphericalHarmonicsToRgb`）
 * 逐字对应，改动时两处必须同步。
 */

/** `sqrt(1/(4π))`，与 `linalg.ts: SQRT4PI` 同值。 */
const SH_C0 = Math.sqrt(1 / (4 * Math.PI))

/** 与 `colorspace.ts: SRGB_THRESHOLD` 同值。 */
const SRGB_THRESHOLD = 0.04045

/**
 * `f_dc_*`（sRGB 域 degree-0 SH）就地转成线性 RGB，写入 `out`。
 *
 * @param shDc `[N*3]` PLY 里的 `f_dc_0..2` 交错数组。
 * @param out  `[N*3]` 输出（可与 `shDc` 同一块内存）。
 */
export function shDcToLinearRgb(shDc: Float32Array, out: Float32Array): void {
  for (let i = 0; i < shDc.length; i++) {
    // 1) SH -> sRGB；越界值先夹到 [0,1]（公开渲染器输出 8bit 时同样会夹）
    let x = shDc[i] * SH_C0 + 0.5
    x = x < 0 ? 0 : x > 1 ? 1 : x
    // 2) sRGB -> linear
    out[i] = x <= SRGB_THRESHOLD ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
  }
}
