# PrismSplat

PrismSplat is the Metal Gaussian splat renderer from Prism. It culls splats against the camera frustum, sorts the visible splats by depth on the GPU, and renders them with indirect draws. A screen-space path composites fine and coarse tiles front to back and falls back to the indexed renderer when the tile index exceeds its capacity.

- Load binary little-endian Gaussian splat PLY files with float32 fields, including LZFSE-compressed PLY at a `.lzfse` URL.
- Render one view or a two-view texture array. The indexed path also supports rasterization-rate maps and optional geometry LOD.
- Import spherical harmonic degrees 0 through 3. The renderer uses the DC and first-order coefficients.
- Use Swift 6 with Metal on Apple silicon, targeting macOS 15, iOS 18, or visionOS 2 and later.

## Add it to a project

In Xcode, add `https://github.com/ferdinandl007/PrismSplat.git` in **File → Add Package Dependencies**. Select the **PrismSplat** library and version **0.1.0** or later.

For Swift Package Manager, add the package and list its product in your target’s dependencies.

```swift
.package(url: "https://github.com/ferdinandl007/PrismSplat.git", from: "0.1.0")
```

```swift
.product(name: "PrismSplat", package: "PrismSplat")
```

## Render a scene

Load the capture and create the renderer once. Use the same color format as your target texture. The package bundles the Metal shader sources and compiles the selected pipelines when the renderer is initialized.

```swift
import Metal
import PrismSplat
import QuartzCore
import simd

func makeRenderer(device: any MTLDevice, sceneURL: URL) async throws -> MetalSplatRenderer {
    let capture = try await SplatFile.read(from: sceneURL, limit: 400_000)
    return try MetalSplatRenderer(device: device, capture: capture, color: .bgra8Unorm_srgb)
}

func drawFrame(
    renderer: MetalSplatRenderer,
    queue: any MTLCommandQueue,
    drawable: any CAMetalDrawable,
    projection: simd_float4x4,
    view: simd_float4x4
) throws {
    let texture = drawable.texture
    let viewport = MTLViewport(
        originX: 0, originY: 0,
        width: Double(texture.width), height: Double(texture.height),
        znear: 0, zfar: 1
    )
    let eye = MetalSplatRenderer.Eye(viewport: viewport, projection: projection, view: view)
    guard let commands = queue.makeCommandBuffer() else {
        throw MetalSplatRenderer.RendererError.allocation
    }
    try renderer.render(eyes: [eye], color: texture, on: commands)
    commands.present(drawable)
    commands.commit()
}
```

The view matrix transforms the PLY’s coordinates into a camera space looking down −Z. Use a Metal perspective projection whose clip-space depth spans 0 to 1. Encode calls serially on the same Metal command queue for each renderer. Start with the default point limit and increase it to suit your memory and frame budget.

For layered stereo, initialize with `maxViews: 2` and supply a two-slice render target with two eyes. For separate targets, call `encodeSort(eyes:on:)` for both eyes and then `draw(eyes:color:on:)` with one eye and its own target at a time. Configure a matching depth attachment format if you need a depth texture. The imageblock depth path uses reversed Z.

For screen-space tiles, initialize the base renderer with `screenTiles: true` and create a `MetalSplatRenderer.ScreenTileRenderer(base:)`. Its `render(eye:color:on:)` takes one full-size viewport and a 2D color target. Set `referenceLimit` on creation to bound the tile-index memory.

## Test

Run `swift test` on an Apple silicon Mac. The tests generate their own PLY data, validate parsing, and read back GPU output from both draw paths.

## License

PrismSplat is available under the [MIT License](LICENSE). See [NOTICE](NOTICE) for the parts adapted from Sean Cier’s MetalSplatter.
