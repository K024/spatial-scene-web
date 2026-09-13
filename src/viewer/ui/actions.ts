/**
 * 载入入口（UI 动作，非状态）。
 *
 * 不挂隐藏 `<input>`：文件选择只在点击时发生一次，即时建一个 input、用完即弃，
 * 比在组件树里维护 ref + 手动 `reset value` 更短、也不会被 React Compiler 的
 * memo 化影响。
 */

import { loadGlbFile, loadGlbUrl, SAMPLE_URL } from "../../store/viewer.ts"

/** 打开系统文件选择框；选中后交给 store。 */
export function pickGlbFile(): void {
  const input = document.createElement("input")
  input.type = "file"
  input.accept = ".glb,model/gltf-binary,application/octet-stream"
  input.addEventListener("change", () => {
    const file = input.files?.[0]
    if (file) void loadGlbFile(file)
  })
  input.click()
}

/** 载入内置样例（`public/models/sample.glb`）。不存在时会走到失败态并给出提示。 */
export function loadSampleGlb(): void {
  void loadGlbUrl(SAMPLE_URL, "sample.glb")
}
