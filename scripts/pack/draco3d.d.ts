/**
 * `draco3d` 没有自带类型声明（也没有 `@types/draco3d`）。
 *
 * 我们只用到它的两个工厂：编码器（写 GLB 时压缩几何）与解码器（读回校验）。
 * 返回值原样交给 `NodeIO.registerDependencies`，所以类型用 `unknown` 即可。
 */
declare module "draco3d" {
  export function createEncoderModule(): Promise<unknown>
  export function createDecoderModule(): Promise<unknown>
}
