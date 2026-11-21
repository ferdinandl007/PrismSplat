import Foundation
import Metal
import PrismSplat
import simd
import Testing

@Test(arguments: [0, 1, 3])
func readsGaussianPLYFromSlicedData(degree: Int) throws {
    let ply = fixture(degree: degree)
    var prefixed = Data([0, 0, 0])
    prefixed.append(ply)
    let capture = try SplatFile.decode(prefixed.dropFirst(3), limit: 1, precision: .float32)
    #expect(capture.count == 1)
    #expect(capture.degree == degree)
    #expect(throws: SplatFile.ImportError.truncated) {
        try SplatFile.decode(Data(ply.dropLast()))
    }
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil), arguments: [nil, 1, 65_536] as [Int?])
func rendersDepthSortedSplats(tileReferenceLimit: Int?) throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    let capture = try SplatFile.decode(fixture())
    let renderer = try MetalSplatRenderer(device: device, capture: capture, color: .rgba8Unorm,
                                         screenTiles: tileReferenceLimit != nil)
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 64, height: 64,
                                                             mipmapped: false)
    descriptor.usage = .renderTarget
    descriptor.storageMode = .shared
    let texture = try #require(device.makeTexture(descriptor: descriptor))
    let commands = try #require(queue.makeCommandBuffer())
    if let tileReferenceLimit {
        let tiles = try MetalSplatRenderer.ScreenTileRenderer(base: renderer, referenceLimit: tileReferenceLimit)
        try tiles.render(eye: fixtureEye, color: texture, on: commands)
    } else {
        try renderer.render(eyes: [fixtureEye], color: texture, on: commands)
    }
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { throw error }
    #expect(commands.status == .completed)

    var center = [UInt8](repeating: 0, count: 4)
    texture.getBytes(&center, bytesPerRow: 4, from: MTLRegionMake2D(32, 32, 1, 1), mipmapLevel: 0)
    #expect(center[0] > 100)
    #expect(center[2] > 0)
    #expect(center[0] > center[2])
}

@Test(.enabled(if: MTLCreateSystemDefaultDevice()?.supportsVertexAmplificationCount(2) == true))
func rendersBothTextureSlices() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let queue = try #require(device.makeCommandQueue())
    let capture = try SplatFile.decode(fixture(degree: 3))
    let renderer = try MetalSplatRenderer(device: device, capture: capture, color: .rgba8Unorm, maxViews: 2)
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 64, height: 64,
                                                             mipmapped: false)
    descriptor.textureType = .type2DArray
    descriptor.arrayLength = 2
    descriptor.usage = .renderTarget
    descriptor.storageMode = .shared
    let texture = try #require(device.makeTexture(descriptor: descriptor))
    let commands = try #require(queue.makeCommandBuffer())
    try renderer.render(eyes: [fixtureEye, fixtureEye], color: texture, on: commands)
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error { throw error }

    for slice in 0..<2 {
        var center = [UInt8](repeating: 0, count: 4)
        texture.getBytes(&center, bytesPerRow: 4, bytesPerImage: 4, from: MTLRegionMake2D(32, 32, 1, 1),
                         mipmapLevel: 0, slice: slice)
        #expect(center[0] > 100)
        #expect(center[2] > 0)
        #expect(center[0] > center[2])
    }
}

private var fixtureEye: MetalSplatRenderer.Eye {
    let viewport = MTLViewport(originX: 0, originY: 0, width: 64, height: 64, znear: 0, zfar: 1)
    let projection = simd_float4x4(columns: ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.02, -1], [0, 0, 0.2, 0]))
    return MetalSplatRenderer.Eye(viewport: viewport, projection: projection, view: matrix_identity_float4x4)
}

private func fixture(degree: Int = 0) -> Data {
    let restCount = 3 * ((degree + 1) * (degree + 1) - 1)
    let fields = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                  "opacity", "f_dc_0", "f_dc_1", "f_dc_2"] + (0..<restCount).map { "f_rest_\($0)" }
    let header = ["ply", "format binary_little_endian 1.0", "element vertex 2"]
        + fields.map { "property float \($0)" } + ["end_header", ""]
    var data = Data(header.joined(separator: "\n").utf8)
    // Put the near red point first in the file to exercise the GPU's sort.
    let points: [[Float]] = [
        [0, 0, -2, -1.5, -1.5, -1.5, 1, 0, 0, 0, 1.4, 1.7, -1.7, -1.7],
        [0, 0, -3, -1.5, -1.5, -1.5, 1, 0, 0, 0, 1.4, -1.7, -1.7, 1.7],
    ]
    for point in points {
        for channel in point + Array(repeating: 0, count: restCount) {
            var bits = channel.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
    }
    return data
}
