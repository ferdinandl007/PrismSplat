import Foundation
import Metal
import simd

/// Renders Gaussian splats using GPU frustum culling, depth sorting and indirect draws.
/// Encode work serially on one command queue per renderer.
public final class MetalSplatRenderer {
    public struct Eye {
        public var viewport: MTLViewport
        public var projection: simd_float4x4
        public var view: simd_float4x4

        public init(viewport: MTLViewport, projection: simd_float4x4, view: simd_float4x4) {
            self.viewport = viewport
            self.projection = projection
            self.view = view
        }
    }

    public struct ScreenTileConfiguration {
        var fine = 16
        var coarse = 128
        var batch = 32
        var fineLimit = 24
        var cutoffExponent = 12
        var exactBounds = true
        var specializeThreadgroup = false

        public init(fine: Int = 16, coarse: Int = 128, batch: Int = 32, fineLimit: Int = 24,
                    cutoffExponent: Int = 12, exactBounds: Bool = true, specializeThreadgroup: Bool = false) {
            self.fine = fine
            self.coarse = coarse
            self.batch = batch
            self.fineLimit = fineLimit
            self.cutoffExponent = cutoffExponent
            self.exactBounds = exactBounds
            self.specializeThreadgroup = specializeThreadgroup
        }

        var threads: Int { fine * fine / 4 }

        var supported: Bool {
            [8, 16, 32].contains(fine) && [64, 128, 256].contains(coarse)
                && (1...threads).contains(batch) && (1...32).contains(fineLimit)
                && (0...20).contains(cutoffExponent)
        }

        var shaderMacros: [String: NSObject] {
            [
                "PRISM_TILE_SIZE": NSNumber(value: fine),
                "PRISM_COARSE_TILE_SIZE": NSNumber(value: coarse),
                "PRISM_TILE_BATCH": NSNumber(value: batch),
                "PRISM_FINE_REFERENCE_LIMIT": NSNumber(value: fineLimit),
                "PRISM_CUTOFF_EXPONENT": NSNumber(value: cutoffExponent),
                "PRISM_EXACT_TILE_BOUNDS": NSNumber(value: exactBounds ? 1 : 0),
            ]
        }
    }

    public enum RendererError: LocalizedError {
        case unsupportedDevice
        case allocation
        case invalidCapture
        case invalidConfiguration
        case shader(String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedDevice: "The graphics processor does not support this Metal renderer."
            case .allocation: "Could not allocate the Metal render resources."
            case .invalidCapture: "The Gaussian splat capture has an unsupported layout or point count."
            case .invalidConfiguration: "The views, render targets, or renderer options are incompatible."
            case .shader(let name): "Could not load the Metal shader \(name)."
            }
        }
    }

    private struct ViewUniforms {
        var projection: simd_float4x4
        var view: simd_float4x4
        var camera: SIMD4<Float>
        var screen: SIMD2<UInt32>
        var focalX: Float
        var focalY: Float
        var tanHalfFovX: Float
        var tanHalfFovY: Float
        var unused: UInt32 = 0
        var count: UInt32 = 0
        var indexedCount: UInt32 = 0
        var padding: (UInt32, UInt32, UInt32) = (0, 0, 0)

        init(_ eye: Eye) {
            projection = eye.projection
            view = eye.view
            camera = eye.view.inverse.columns.3
            screen = [UInt32(eye.viewport.width), UInt32(eye.viewport.height)]
            focalX = Float(screen.x) * projection.columns.0.x / 2
            focalY = Float(screen.y) * projection.columns.1.y / 2
            tanHalfFovX = 1 / projection.columns.0.x
            tanHalfFovY = 1 / projection.columns.1.y
        }
    }

    private struct CullView {
        var clipFromCapture: simd_float4x4
        var ellipseMargin: SIMD2<Float>
        var pixelMargin: SIMD2<Float>

        init(_ eye: Eye) {
            clipFromCapture = eye.projection * eye.view
            let focal = SIMD2(eye.projection.columns.0.x, eye.projection.columns.1.y)
            let oblique = 1 + simd_abs(SIMD2(eye.projection.columns.2.x, eye.projection.columns.2.y))
            let marginSquared = focal * focal + oblique * oblique
            ellipseMargin = SIMD2(sqrt(marginSquared.x), sqrt(marginSquared.y))
            // The rasterizer's low-pass filter adds a small footprint even to subpixel splats.
            pixelMargin = 4 / SIMD2(Float(eye.viewport.width), Float(eye.viewport.height))
        }
    }

    private struct SortUniforms {
        var eyes: (CullView, CullView)
        var origin: SIMD4<Float>
        var forward: SIMD4<Float>
        var count: UInt32
        var eyeCount: UInt32
        var padding: (UInt32, UInt32) = (0, 0)
    }

    private struct Point {
        // The upstream Splat record uses a packed float3 and ten half values.
        var position: MTLPackedFloat3
        var color: (Float16, Float16, Float16, Float16)
        var covarianceA: (Float16, Float16, Float16)
        var covarianceB: (Float16, Float16, Float16)
    }

    private static let width = 128
    private static let splatsPerInstance = 256
    private static let sortGridOffset = 16
    private static let scanGridOffset = 32
    private static let drawOffset = 48

    private let library: any MTLLibrary
    private let screenTileConfiguration: ScreenTileConfiguration
    private let screenTiles: Bool
    private let colorFormat: MTLPixelFormat
    private let depthFormat: MTLPixelFormat
    private let maxViews: Int
    private let geometryLOD: GeometryLOD?
    private let count: Int
    private let groups: Int
    private let tiles: Int
    private let points: any MTLBuffer
    private let harmonics: any MTLBuffer
    private let frameOwner = UUID()
    private let temporary: any MTLBuffer
    private let dense: any MTLBuffer
    private let scatter: any MTLBuffer
    private let counts: any MTLBuffer
    private let tileCounts: any MTLBuffer
    private let totals: any MTLBuffer
    private let state: any MTLBuffer
    private let indices: any MTLBuffer
    private let cull: any MTLComputePipelineState
    private let scanGroups: any MTLComputePipelineState
    private let scanTiles: any MTLComputePipelineState
    private let compact: any MTLComputePipelineState
    private let histogram: any MTLComputePipelineState
    private let radixScatter: any MTLComputePipelineState
    private let renderPipeline: any MTLRenderPipelineState
    private let initializeTile: (any MTLRenderPipelineState)?
    private let resolveTile: (any MTLRenderPipelineState)?
    private let depthState: any MTLDepthStencilState

    /// Creates the GPU resources and compiles the shaders bundled with the package.
    public init(device: any MTLDevice, capture: SplatFile, color: MTLPixelFormat = .bgra8Unorm_srgb,
                depth: MTLPixelFormat = .invalid, maxViews: Int = 1, tileDepth: Bool = false,
                screenTiles: Bool = false, screenTileConfiguration: ScreenTileConfiguration = .init(),
                geometryLOD: Bool = false, spatialLOD: Bool = true) throws {
        guard capture.count > 0, capture.count <= 2 * Self.width * Self.width * Self.width,
              MemoryLayout<Point>.stride == 32, MemoryLayout<ViewUniforms>.stride == 192 else {
            throw RendererError.invalidCapture
        }
        guard (1...2).contains(maxViews), color != .invalid, !tileDepth || depth != .invalid,
              !screenTiles || (screenTileConfiguration.supported && !tileDepth && depth == .invalid) else {
            throw RendererError.invalidConfiguration
        }
        guard device.supportsVertexAmplificationCount(maxViews) else { throw RendererError.unsupportedDevice }
        guard let shaderDirectory = Bundle.module.url(forResource: "Shaders", withExtension: nil) else {
            throw RendererError.shader("Shaders")
        }
        self.count = capture.count
        self.colorFormat = color
        self.depthFormat = depth
        self.maxViews = maxViews
        self.screenTiles = screenTiles
        self.screenTileConfiguration = screenTileConfiguration
        groups = (count + Self.width - 1) / Self.width
        tiles = (groups + Self.width - 1) / Self.width

        func allocate(_ bytes: Int, _ label: String, storage: MTLResourceOptions = .storageModePrivate) throws -> any MTLBuffer {
            guard let buffer = device.makeBuffer(length: bytes, options: storage) else { throw RendererError.allocation }
            buffer.label = label
            return buffer
        }

        let records = count + (geometryLOD ? GeometryLOD.levelCounts(count).reduce(0, +) : 0)
        points = try allocate(records * MemoryLayout<Point>.stride, "MetalSplatter points", storage: .storageModeShared)
        harmonics = try allocate(records * 9 * MemoryLayout<Float16>.stride, "MetalSplatter SH1", storage: .storageModeShared)
        temporary = try allocate(count * 8, "Frustum candidates")
        dense = try allocate(count * 8, "Radix keys A")
        scatter = try allocate(count * 8, "Radix keys B")
        counts = try allocate(groups * 16 * 4, "Workgroup prefix")
        tileCounts = try allocate(tiles * 16 * 4, "Tile prefix")
        totals = try allocate(16 * 4, "Radix bucket totals")
        state = try allocate(80, "GPU dispatch and draw counts", storage: .storageModeShared)
        indices = try allocate(Self.splatsPerInstance * 6 * 2, "Splat quad indices", storage: .storageModeShared)

        let shaderNames = ["ShaderCommon.h", "SplatProcessing.msl", "PrismSorting.msl", "PrismDraw.msl"]
            + (tileDepth ? ["TileBlend.msl"] : [])
            + (screenTiles ? ["PrismScreenTiles.msl"] : [])
            + (geometryLOD ? ["PrismGeometryLOD.msl"] : [])
        let source = try shaderNames.map {
            try String(contentsOf: shaderDirectory.appendingPathComponent($0), encoding: .utf8)
        }.joined(separator: "\n")
        let options = MTLCompileOptions()
        var macros = screenTileConfiguration.shaderMacros
        macros["PRISM_GEOMETRY_LOD"] = NSNumber(value: geometryLOD)
        macros["PRISM_SPATIAL_LOD"] = NSNumber(value: geometryLOD && spatialLOD)
        macros["PRISM_FLAT_SPLAT_COLOR"] = NSNumber(value: capture.degree == 0)
        options.preprocessorMacros = macros
        let library = try device.makeLibrary(source: source, options: options)
        self.library = library

        func function(_ name: String) throws -> any MTLFunction {
            guard let shader = library.makeFunction(name: name) else { throw RendererError.shader(name) }
            return shader
        }
        func compute(_ name: String) throws -> any MTLComputePipelineState {
            let pipeline = try device.makeComputePipelineState(function: function(name))
            guard pipeline.threadExecutionWidth == 32,
                  pipeline.maxTotalThreadsPerThreadgroup >= Self.width else { throw RendererError.unsupportedDevice }
            return pipeline
        }
        cull = try compute("prismCull")
        scanGroups = try compute("prismScanGroups")
        scanTiles = try compute("prismScanTiles")
        compact = try compute("prismCompact")
        histogram = try compute("prismHistogram")
        radixScatter = try compute("prismScatter")

        let render = MTLRenderPipelineDescriptor()
        render.vertexFunction = try function("prismSplatVertex")
        render.fragmentFunction = try function(tileDepth ? "multiStageSplatFragmentShader" : "prismSplatFragment")
        render.maxVertexAmplificationCount = maxViews
        render.depthAttachmentPixelFormat = depth
        render.colorAttachments[0].pixelFormat = color
        if !tileDepth {
            let target = render.colorAttachments[0]!
            target.isBlendingEnabled = true
            target.sourceRGBBlendFactor = .one
            target.destinationRGBBlendFactor = .oneMinusSourceAlpha
            target.sourceAlphaBlendFactor = .one
            target.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        }
        renderPipeline = try device.makeRenderPipelineState(descriptor: render)

        if tileDepth {
            let tile = MTLTileRenderPipelineDescriptor()
            tile.tileFunction = try function("initializeFragmentStore")
            tile.threadgroupSizeMatchesTileSize = true
            tile.colorAttachments[0].pixelFormat = color
            initializeTile = try device.makeRenderPipelineState(tileDescriptor: tile, options: [], reflection: nil)
            render.vertexFunction = try function("postprocessVertexShader")
            render.fragmentFunction = try function("postprocessFragmentShader")
            resolveTile = try device.makeRenderPipelineState(descriptor: render)
        } else {
            initializeTile = nil
            resolveTile = nil
        }

        let descriptor = MTLDepthStencilDescriptor()
        descriptor.depthCompareFunction = .always
        descriptor.isDepthWriteEnabled = depth != .invalid
        guard let depthState = device.makeDepthStencilState(descriptor: descriptor) else { throw RendererError.allocation }
        self.depthState = depthState

        try Self.encode(capture, into: points, harmonics: harmonics)
        self.geometryLOD = geometryLOD ? try GeometryLOD(device: device, library: library, count: count,
                                                         points: points, harmonics: harmonics, spatial: spatialLOD) : nil
        let index = indices.contents().bindMemory(to: UInt16.self, capacity: Self.splatsPerInstance * 6)
        for point in 0..<Self.splatsPerInstance {
            let corner = UInt16(point * 4)
            let base = point * 6
            index[base] = corner
            index[base + 1] = corner + 1
            index[base + 2] = corner + 2
            index[base + 3] = corner + 1
            index[base + 4] = corner + 2
            index[base + 5] = corner + 3
        }
    }

    private static func encode(_ file: SplatFile, into points: any MTLBuffer, harmonics: any MTLBuffer) throws {
        let targets = points.contents().bindMemory(to: Point.self, capacity: file.count)
        let sh = harmonics.contents().bindMemory(to: Float16.self, capacity: file.count * 9)
        let isCompact = file.precision == .compact
        try file.attributes.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            for index in 0..<file.count {
                if index.isMultiple(of: 4_096) { try Task.checkCancellation() }
                let row = index * file.stride
                func value(_ field: Int) -> Float {
                    let offset = row + file.precision.offset(of: field)
                    if isCompact, field >= 3 { return Float(bytes.loadUnaligned(fromByteOffset: offset, as: Float16.self)) }
                    return bytes.loadUnaligned(fromByteOffset: offset, as: Float.self)
                }
                let q = simd_quatf(ix: value(7), iy: value(8), iz: value(9), r: value(6))
                let scale = simd_float3x3(diagonal: [exp(value(3)), exp(value(4)), exp(value(5))])
                let transform = simd_float3x3(simd_normalize(q)) * scale
                let cov = transform * transform.transpose
                let alpha = file.usesLinearOpacity ? value(10) : 1 / (1 + exp(-value(10)))
                func half(_ value: Float) -> Float16 { Float16(max(-65_504, min(65_504, value))) }
                targets[index] = Point(position: MTLPackedFloat3Make(value(0), value(1), value(2)),
                    color: (half(value(11)), half(value(12)), half(value(13)), half(alpha)),
                    covarianceA: (half(cov[0, 0]), half(cov[0, 1]), half(cov[0, 2])),
                    covarianceB: (half(cov[1, 1]), half(cov[1, 2]), half(cov[2, 2])))
                for field in 0..<9 { sh[index * 9 + field] = file.degree > 0 ? half(value(14 + field)) : 0 }
            }
        }
    }

    /// An immutable upload for one animation frame. Retain only the frames needed for playback.
    public struct PreparedFrame: @unchecked Sendable {
        fileprivate let owner: UUID
        fileprivate let points: any MTLBuffer
        fileprivate let harmonics: any MTLBuffer

        public var byteCount: Int { points.length + harmonics.length }
    }

    /// Returns a worker-safe encoder for captures with the renderer's original point count.
    public func framePreparation() throws -> @Sendable (SplatFile) throws -> PreparedFrame {
        guard geometryLOD == nil else { throw RendererError.invalidConfiguration }
        let device = points.device
        let pointsLength = points.length
        let harmonicsLength = harmonics.length
        let owner = frameOwner
        let fixedCount = count
        return { capture in
            guard capture.count == fixedCount else { throw RendererError.invalidCapture }
            guard let points = device.makeBuffer(length: pointsLength, options: .storageModeShared),
                  let harmonics = device.makeBuffer(length: harmonicsLength, options: .storageModeShared) else {
                throw RendererError.allocation
            }
            try Self.encode(capture, into: points, harmonics: harmonics)
            return PreparedFrame(owner: owner, points: points, harmonics: harmonics)
        }
    }

    /// Enqueues a geometry change before sorting and drawing on the same command queue.
    public func encodeFrame(_ frame: PreparedFrame, on command: any MTLCommandBuffer) throws {
        guard frame.owner == frameOwner else { throw RendererError.invalidConfiguration }
        guard let blit = command.makeBlitCommandEncoder() else { throw RendererError.allocation }
        blit.label = "Splat frame upload"
        blit.copy(from: frame.points, sourceOffset: 0, to: points, destinationOffset: 0, size: points.length)
        blit.copy(from: frame.harmonics, sourceOffset: 0, to: harmonics, destinationOffset: 0, size: harmonics.length)
        blit.endEncoding()
    }

    /// Call on a single command queue. Tracked Metal buffers keep later frames
    /// from overwriting the order while a previous frame is rasterizing.
    public func encodeSort(eyes: [Eye], on command: any MTLCommandBuffer, rates: [EyeRate] = [],
                           maxPeripheralRadius: Float = 12, outerLevels: Bool = true, finestLevel: Int = 0) throws {
        guard (1...2).contains(eyes.count), eyes.allSatisfy(Self.validViewport) else {
            throw RendererError.invalidConfiguration
        }
        guard let first = eyes.first, var encoder = command.makeComputeCommandEncoder() else { throw RendererError.allocation }
        let level = geometryLOD == nil ? 0 : min(3, max(0, finestLevel))
        let stride = 1 << (3 * level)
        let candidates = (count + stride - 1) / stride
        let activeGroups = (candidates + Self.width - 1) / Self.width
        let activeTiles = (activeGroups + Self.width - 1) / Self.width
        let second = eyes.last ?? first
        let firstPose = first.view.inverse
        let secondPose = second.view.inverse
        let cameraForward = (firstPose.columns.2 + secondPose.columns.2) * -0.5
        var uniforms = SortUniforms(eyes: (CullView(first), CullView(second)),
            origin: (firstPose.columns.3 + secondPose.columns.3) * 0.5, forward: cameraForward,
            count: UInt32(count), eyeCount: UInt32(eyes.count))
        let lanes = MTLSize(width: Self.width, height: 1, depth: 1)
        encoder.label = "Splat frustum culling and compaction"

        func scalar(_ value: UInt32, _ index: Int) {
            var value = value
            encoder.setBytes(&value, length: 4, index: index)
        }
        func bind(_ pipeline: any MTLComputePipelineState, _ buffers: [any MTLBuffer]) {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffers(buffers.map(Optional.some), offsets: Array(repeating: 0, count: buffers.count),
                               range: 0..<buffers.count)
        }
        func run(_ width: Int, _ height: Int = 1) {
            encoder.dispatchThreadgroups(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: lanes)
            encoder.memoryBarrier(scope: .buffers)
        }
        func runIndirect(_ offset: Int) {
            encoder.dispatchThreadgroups(indirectBuffer: state, indirectBufferOffset: offset, threadsPerThreadgroup: lanes)
            encoder.memoryBarrier(scope: .buffers)
        }
        func prefix(radix: Bool) {
            bind(scanGroups, [counts, tileCounts])
            scalar(UInt32(radix ? groups : activeGroups), 2)
            scalar(UInt32(radix ? tiles : activeTiles), 3)
            encoder.setBuffer(state, offset: 0, index: 4)
            scalar(radix ? 1 : 0, 5)
            if radix { runIndirect(Self.scanGridOffset) } else { run(activeTiles) }

            bind(scanTiles, [tileCounts, totals, state])
            scalar(UInt32(radix ? tiles : activeTiles), 3)
            scalar(radix ? 1 : 0, 4)
            run(radix ? 16 : 1)
        }

        if let geometryLOD {
            uniforms.padding.0 = geometryLOD.select(eyes: eyes, rates: rates, radius: maxPeripheralRadius,
                                                    outerLevels: outerLevels, finestLevel: level, on: encoder) ? 1 : 0
            uniforms.padding.1 = UInt32(level << 8) | (outerLevels ? 1 : 0)
        }
        bind(cull, [points, temporary, counts])
        if let geometryLOD {
            encoder.setBuffer(geometryLOD.decisions, offset: 0, index: 4)
            if let ids = geometryLOD.orderedLeaves { encoder.setBuffer(ids, offset: count * 4, index: 5) }
        }
        withUnsafeBytes(of: uniforms) { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        run(activeGroups)
        prefix(radix: false)
        bind(compact, [temporary, dense, counts, tileCounts, state])
        scalar(UInt32(activeGroups), 5)
        run(activeGroups)
        encoder.endEncoding()

        guard let sorter = command.makeComputeCommandEncoder() else { throw RendererError.allocation }
        encoder = sorter
        encoder.label = "Splat GPU radix depth sort"
        for pass in 0..<8 {
            let input = pass.isMultiple(of: 2) ? dense : scatter
            let output = pass.isMultiple(of: 2) ? scatter : dense
            let shift = UInt32(pass * 4)
            bind(histogram, [input, counts, state])
            scalar(UInt32(groups), 3)
            scalar(shift, 4)
            runIndirect(Self.sortGridOffset)
            prefix(radix: true)
            bind(radixScatter, [input, output, counts, tileCounts, totals, state])
            scalar(UInt32(groups), 6)
            scalar(UInt32(tiles), 7)
            scalar(shift, 8)
            runIndirect(Self.sortGridOffset)
        }
        encoder.endEncoding()
    }

    /// Encodes culling, sorting and drawing into a caller-owned command buffer.
    public func render(eyes: [Eye], color: any MTLTexture, depth: (any MTLTexture)? = nil,
                       rate: (any MTLRasterizationRateMap)? = nil, rates: [EyeRate] = [],
                       clearColor: MTLClearColor = .init(red: 0, green: 0, blue: 0, alpha: 1),
                       on command: any MTLCommandBuffer) throws {
        try encodeSort(eyes: eyes, on: command, rates: rates)
        try draw(eyes: eyes, color: color, depth: depth, rate: rate, clearColor: clearColor, on: command)
    }

    /// Draws after `encodeSort`. A two-eye draw uses two slices of a texture array.
    public func draw(eyes: [Eye], color: any MTLTexture, depth: (any MTLTexture)? = nil,
                     rate: (any MTLRasterizationRateMap)? = nil,
                     clearColor: MTLClearColor = .init(red: 0, green: 0, blue: 0, alpha: 1),
                     on command: any MTLCommandBuffer,
                     overlay: ((any MTLRenderCommandEncoder) -> Void)? = nil) throws {
        guard let first = eyes.first, eyes.count <= maxViews, eyes.allSatisfy(Self.validViewport),
              color.pixelFormat == colorFormat, color.sampleCount == 1, color.arrayLength >= eyes.count,
              (depth?.pixelFormat ?? .invalid) == depthFormat,
              depth.map({ $0.sampleCount == 1 && $0.width == color.width && $0.height == color.height
                          && $0.arrayLength >= eyes.count }) ?? true else {
            throw RendererError.invalidConfiguration
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = clearColor
        if let depth {
            pass.depthAttachment.texture = depth
            pass.depthAttachment.loadAction = .clear
            pass.depthAttachment.storeAction = .store
            pass.depthAttachment.clearDepth = 0
        }
        pass.rasterizationRateMap = rate
        pass.renderTargetArrayLength = eyes.count
        if let initializeTile {
            pass.tileWidth = 16
            pass.tileHeight = 16
            pass.imageblockSampleLength = initializeTile.imageblockSampleLength
        }
        guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { throw RendererError.allocation }
        encoder.label = "Splat projection and blending"
        encoder.setViewports(eyes.map(\.viewport))
        if eyes.count > 1 {
            let mappings = eyes.indices.map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            mappings.withUnsafeBufferPointer { encoder.setVertexAmplificationCount(eyes.count, viewMappings: $0.baseAddress) }
        }
        if let initializeTile {
            encoder.setRenderPipelineState(initializeTile)
            encoder.dispatchThreadsPerTile(MTLSize(width: 16, height: 16, depth: 1))
        }

        encodeQuads(first: first, last: eyes.last ?? first, arguments: state, on: encoder)
        if let resolveTile {
            encoder.setRenderPipelineState(resolveTile)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        overlay?(encoder)
        encoder.endEncoding()
    }

    private static func validViewport(_ eye: Eye) -> Bool {
        let viewport = eye.viewport
        let limit = Double(UInt32.max)
        return (1...limit).contains(viewport.width) && (1...limit).contains(viewport.height)
    }

    private func encodeQuads(first: Eye, last: Eye, arguments: any MTLBuffer, on encoder: any MTLRenderCommandEncoder) {
        encoder.setDepthStencilState(depthState)
        let uniforms = (ViewUniforms(first), ViewUniforms(last))
        encoder.setRenderPipelineState(renderPipeline)
        withUnsafeBytes(of: uniforms) { encoder.setVertexBytes($0.baseAddress!, length: $0.count, index: 0) }
        encoder.setVertexBuffer(points, offset: 0, index: 1)
        encoder.setVertexBuffer(dense, offset: 0, index: 2)
        encoder.setVertexBuffer(state, offset: 0, index: 3)
        encoder.setVertexBuffer(harmonics, offset: 0, index: 4)
        encoder.drawIndexedPrimitives(type: .triangle, indexType: .uint16, indexBuffer: indices,
                                      indexBufferOffset: 0, indirectBuffer: arguments, indirectBufferOffset: Self.drawOffset)
    }
}

extension MetalSplatRenderer {
    /// Composites one eye front to back and uses the indexed draw path on tile-index overflow.
    public final class ScreenTileRenderer {
        private struct Grid {
            let size: SIMD2<UInt32>
            let fine: SIMD2<UInt32>
            let coarse: SIMD2<UInt32>
            let referenceCapacity: UInt32
            let padding: UInt32 = 0
        }

        var fineSize: Int { base.screenTileConfiguration.fine }
        var coarseSize: Int { base.screenTileConfiguration.coarse }
        private let base: MetalSplatRenderer
        private let device: any MTLDevice
        private let project: any MTLComputePipelineState
        private let scan: any MTLComputePipelineState
        private let emit: any MTLComputePipelineState
        private let clearRanges: any MTLComputePipelineState
        private let findRanges: any MTLComputePipelineState
        private let tileHistogram: any MTLComputePipelineState
        private let tileScatter: any MTLComputePipelineState
        private let blend: any MTLComputePipelineState
        private let resolve: any MTLRenderPipelineState

        private let capacity: Int
        private let groups: Int
        private let chunks: Int
        private let projected: any MTLBuffer
        private let footprints: any MTLBuffer
        private let offsets: any MTLBuffer
        private let keys: any MTLBuffer
        private let scratch: any MTLBuffer
        private let ranks: any MTLBuffer
        private let rankScratch: any MTLBuffer
        private let counts: any MTLBuffer
        private let chunkCounts: any MTLBuffer
        private let totals: any MTLBuffer
        private let state: any MTLBuffer
        private var ranges: (any MTLBuffer)?
        private var linearTarget: (any MTLTexture)?

        /// The indexed renderer must be initialized with `screenTiles: true`.
        public init(base: MetalSplatRenderer, referenceLimit: Int = 8_388_608) throws {
            guard base.screenTiles, referenceLimit > 0 else { throw RendererError.invalidConfiguration }
            let device = base.library.device
            self.base = base
            self.device = device
            let lanes = MetalSplatRenderer.width
            capacity = max(1, min(referenceLimit, max(base.count * 2, 65_536), 4 * lanes * lanes * lanes))
            groups = (capacity + lanes - 1) / lanes
            chunks = (groups + lanes - 1) / lanes

            func buffer(_ bytes: Int, _ label: String, _ options: MTLResourceOptions = .storageModePrivate) throws -> any MTLBuffer {
                guard let buffer = device.makeBuffer(length: bytes, options: options) else { throw RendererError.allocation }
                buffer.label = label
                return buffer
            }
            projected = try buffer(base.count * 32, "Projected splat conics")
            footprints = try buffer(base.count * 16, "Splat tile footprints")
            offsets = try buffer(base.count * 4, "Offsets within projected workgroups")
            keys = try buffer(capacity * 2, "Tile IDs")
            scratch = try buffer(capacity * 2, "Tile ID sort scratch")
            ranks = try buffer(capacity * 4, "Stable tile depth ranks")
            rankScratch = try buffer(capacity * 4, "Tile rank sort scratch")
            counts = try buffer(max(groups * 16, base.groups) * 4, "Tile reference group prefixes")
            chunkCounts = try buffer(max(chunks * 16, base.tiles) * 4, "Tile reference chunk prefixes")
            totals = try buffer(16 * 4, "Tile radix totals")
            state = try buffer(80, "Tile index metadata and overflow fallback", .storageModeShared)

            func function(_ name: String) throws -> any MTLFunction {
                guard let function = base.library.makeFunction(name: name) else { throw RendererError.shader(name) }
                return function
            }
            func compute(_ name: String) throws -> any MTLComputePipelineState {
                try device.makeComputePipelineState(function: function(name))
            }
            project = try compute("prismTileProject")
            scan = try compute("prismScanTileReferences")
            emit = try compute("prismTileEmit")
            clearRanges = try compute("prismClearTileRanges")
            findRanges = try compute("prismTileRanges")
            tileHistogram = try compute("prismCompactHistogram")
            tileScatter = try compute("prismCompactScatter")
            let config = base.screenTileConfiguration
            let blend: any MTLComputePipelineState
            if config.specializeThreadgroup {
                let descriptor = MTLComputePipelineDescriptor()
                descriptor.computeFunction = try function("prismTileBlend")
                descriptor.maxTotalThreadsPerThreadgroup = config.threads
                descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = config.threads.isMultiple(of: 32)
                blend = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
            } else {
                blend = try compute("prismTileBlend")
            }
            guard blend.threadExecutionWidth == 32, blend.maxTotalThreadsPerThreadgroup >= config.threads else {
                throw RendererError.unsupportedDevice
            }
            self.blend = blend
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = try function("prismTileResolveVertex")
            descriptor.fragmentFunction = try function("prismTileResolve")
            descriptor.colorAttachments[0].pixelFormat = base.colorFormat
            resolve = try device.makeRenderPipelineState(descriptor: descriptor)
        }

        private func targets(for color: any MTLTexture, rangeBytes: Int) throws -> (any MTLTexture, any MTLBuffer) {
            if linearTarget?.width != color.width || linearTarget?.height != color.height {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float, width: color.width, height: color.height, mipmapped: false)
                descriptor.storageMode = .private
                descriptor.usage = [.shaderRead, .shaderWrite]
                linearTarget = device.makeTexture(descriptor: descriptor)
                linearTarget?.label = "Linear front-to-back output"
            }
            if (ranges?.length ?? 0) < rangeBytes {
                ranges = device.makeBuffer(length: rangeBytes, options: .storageModePrivate)
                ranges?.label = "Fine and coarse screen tile ranges"
            }
            guard let linearTarget, let ranges else { throw RendererError.allocation }
            return (linearTarget, ranges)
        }

        public func render(eye: Eye, color: any MTLTexture, on command: any MTLCommandBuffer) throws {
            try base.encodeSort(eyes: [eye], on: command)
            try draw(eye: eye, color: color, on: command)
        }

        /// Draws a single eye after the indexed renderer's `encodeSort`.
        public func draw(eye: Eye, color: any MTLTexture, on command: any MTLCommandBuffer) throws {
            guard MetalSplatRenderer.validViewport(eye), color.pixelFormat == base.colorFormat,
                  color.textureType == .type2D, color.sampleCount == 1, eye.viewport.originX == 0,
                  eye.viewport.originY == 0, eye.viewport.width == Double(color.width),
                  eye.viewport.height == Double(color.height) else { throw RendererError.invalidConfiguration }
            let size = SIMD2(UInt32(color.width), UInt32(color.height))
            let fine = (size &+ (UInt32(fineSize) - 1)) / UInt32(fineSize)
            let coarse = (size &+ (UInt32(coarseSize) - 1)) / UInt32(coarseSize)
            let tileCount = fine.x * fine.y + coarse.x * coarse.y
            // Pixel offsets and tile IDs fit in the shader's 16-bit fields.
            guard tileCount <= 65_536 else {
                try base.draw(eyes: [eye], color: color, on: command)
                return
            }
            var grid = Grid(size: size, fine: fine, coarse: coarse, referenceCapacity: UInt32(capacity))
            let (linear, ranges) = try targets(for: color, rangeBytes: Int(tileCount) * 8)
            guard let indexer = command.makeComputeCommandEncoder() else { throw RendererError.allocation }
            indexer.label = "Splat projection and screen tile index"
            let width = MetalSplatRenderer.width
            let lanes = MTLSize(width: width, height: 1, depth: 1)
            func set(_ pipeline: any MTLComputePipelineState, _ buffers: [any MTLBuffer]) {
                indexer.setComputePipelineState(pipeline)
                indexer.setBuffers(buffers.map(Optional.some), offsets: Array(repeating: 0, count: buffers.count),
                                   range: 0..<buffers.count)
            }
            func scalar(_ value: UInt32, at index: Int) {
                var value = value
                indexer.setBytes(&value, length: 4, index: index)
            }
            func direct(_ groups: Int) {
                indexer.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: lanes)
                indexer.memoryBarrier(scope: .buffers)
            }
            func indirect(_ state: any MTLBuffer, _ offset: Int) {
                indexer.dispatchThreadgroups(indirectBuffer: state, indirectBufferOffset: offset, threadsPerThreadgroup: lanes)
                indexer.memoryBarrier(scope: .buffers)
            }
            func scanChunks(radix: Bool, stride: Int) {
                set(scan, [chunkCounts, totals, state])
                scalar(UInt32(stride), at: 3)
                scalar(radix ? 1 : 0, at: 4)
                indexer.setBytes(&grid, length: MemoryLayout<Grid>.size, index: 5)
                indexer.setBuffer(base.state, offset: 0, index: 6)
                direct(radix ? 16 : 1)
            }
            func scanGroups(radix: Bool, stride: Int, chunks: Int, inputState: any MTLBuffer) {
                set(base.scanGroups, [counts, chunkCounts])
                scalar(UInt32(stride), at: 2)
                scalar(UInt32(chunks), at: 3)
                indexer.setBuffer(inputState, offset: 0, index: 4)
                scalar(1, at: 5)
                if radix { indirect(inputState, MetalSplatRenderer.scanGridOffset) } else { direct(chunks) }
            }

            indexer.setComputePipelineState(project)
            indexer.setBuffer(base.points, offset: 0, index: 0)
            indexer.setBuffer(base.harmonics, offset: 0, index: 1)
            indexer.setBuffer(base.dense, offset: 0, index: 2)
            indexer.setBuffer(base.state, offset: 0, index: 3)
            var uniforms = ViewUniforms(eye)
            indexer.setBytes(&uniforms, length: MemoryLayout<ViewUniforms>.stride, index: 4)
            indexer.setBytes(&grid, length: MemoryLayout<Grid>.size, index: 5)
            indexer.setBuffer(projected, offset: 0, index: 6)
            indexer.setBuffer(footprints, offset: 0, index: 7)
            indexer.setBuffer(offsets, offset: 0, index: 8)
            indexer.setBuffer(counts, offset: 0, index: 9)
            indirect(base.state, MetalSplatRenderer.sortGridOffset)
            scanGroups(radix: false, stride: base.groups, chunks: base.tiles, inputState: base.state)
            scanChunks(radix: false, stride: base.tiles)
            set(emit, [footprints, offsets, counts, chunkCounts, base.state, state])
            indexer.setBytes(&grid, length: MemoryLayout<Grid>.size, index: 6)
            indexer.setBuffer(keys, offset: 0, index: 7)
            indexer.setBuffer(ranks, offset: 0, index: 8)
            indirect(base.state, MetalSplatRenderer.sortGridOffset)

            // Four stable digits cover all fine and coarse tile IDs while preserving depth within each list.
            for pass in 0..<4 {
                let input = pass.isMultiple(of: 2) ? keys : scratch
                let output = pass.isMultiple(of: 2) ? scratch : keys
                let shift = UInt32(pass * 4)
                set(tileHistogram, [input, counts, state])
                scalar(UInt32(groups), at: 3)
                scalar(shift, at: 4)
                indirect(state, MetalSplatRenderer.sortGridOffset)
                scanGroups(radix: true, stride: groups, chunks: chunks, inputState: state)
                scanChunks(radix: true, stride: chunks)
                set(tileScatter, [input, output, counts, chunkCounts, totals, state])
                scalar(UInt32(groups), at: 6)
                scalar(UInt32(chunks), at: 7)
                scalar(shift, at: 8)
                indexer.setBuffer(pass.isMultiple(of: 2) ? ranks : rankScratch, offset: 0, index: 9)
                indexer.setBuffer(pass.isMultiple(of: 2) ? rankScratch : ranks, offset: 0, index: 10)
                indirect(state, MetalSplatRenderer.sortGridOffset)
            }

            set(clearRanges, [ranges])
            scalar(tileCount * 2, at: 1)
            direct((Int(tileCount) * 2 + width - 1) / width)
            set(findRanges, [keys, state, ranges])
            indirect(state, MetalSplatRenderer.sortGridOffset)
            indexer.endEncoding()

            guard let blender = command.makeComputeCommandEncoder() else { throw RendererError.allocation }
            blender.label = "Splat front-to-back tile compositing"
            blender.setComputePipelineState(blend)
            blender.setTexture(linear, index: 0)
            blender.setBuffers([projected, ranks, ranges, state], offsets: [0, 0, 0, 0], range: 0..<4)
            blender.setBytes(&grid, length: MemoryLayout<Grid>.size, index: 4)
            blender.dispatchThreadgroups(MTLSize(width: Int(fine.x), height: Int(fine.y), depth: 1),
                threadsPerThreadgroup: MTLSize(width: fineSize / 2, height: fineSize / 2, depth: 1))
            blender.endEncoding()

            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = color
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            guard let encoder = command.makeRenderCommandEncoder(descriptor: pass) else { throw RendererError.allocation }
            encoder.label = "Splat tile resolve and indexed overflow fallback"
            encoder.setRenderPipelineState(resolve)
            encoder.setFragmentTexture(linear, index: 0)
            encoder.setFragmentBuffer(state, offset: 0, index: 0)
            var capacity = grid.referenceCapacity
            encoder.setFragmentBytes(&capacity, length: 4, index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.setViewport(eye.viewport)
            base.encodeQuads(first: eye, last: eye, arguments: state, on: encoder)
            encoder.endEncoding()
        }
    }
}

extension MetalSplatRenderer {
    /// A rasterization-rate map and its slice for one eye, used by geometry LOD.
    public struct EyeRate {
        public let map: any MTLRasterizationRateMap
        public let slice: Int

        public init(map: any MTLRasterizationRateMap, slice: Int) {
            self.map = map
            self.slice = slice
        }
    }

    private final class GeometryLOD {
        private struct View {
            var clip: simd_float4x4
            var viewport: SIMD4<Float>
            var mapSize: SIMD2<Float>
            var focal: SIMD2<Float>
        }
        let decisions: any MTLBuffer
        let orderedLeaves: (any MTLBuffer)?
        private let bounds: any MTLBuffer
        private let densities: any MTLBuffer
        private let sample: any MTLComputePipelineState
        private let classify: any MTLComputePipelineState
        private let levels: [Int]
        private let device: any MTLDevice

        static func levelCounts(_ leaves: Int) -> [Int] {
            let fine = (leaves + 7) / 8
            let medium = (fine + 7) / 8
            return [fine, medium, (medium + 7) / 8]
        }

        init(device: any MTLDevice, library: any MTLLibrary, count: Int,
             points: any MTLBuffer, harmonics: any MTLBuffer, spatial: Bool) throws {
            self.device = device
            levels = Self.levelCounts(count)
            let groups = levels.reduce(0, +)
            func kernel(_ name: String) throws -> any MTLComputePipelineState {
                guard let f = library.makeFunction(name: name) else { throw RendererError.shader(name) }
                return try device.makeComputePipelineState(function: f)
            }
            sample = try kernel("prismSampleRates")
            classify = try kernel("prismSelectParents")
            let build = try kernel("prismMakeParents")
            guard sample.threadExecutionWidth == 32, sample.maxTotalThreadsPerThreadgroup >= 128,
                  let bounds = device.makeBuffer(length: groups * 16, options: .storageModePrivate),
                  let decisions = device.makeBuffer(length: groups, options: .storageModePrivate),
                  let densities = device.makeBuffer(length: 2 * 129 * 8, options: .storageModePrivate),
                  let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder() else { throw RendererError.allocation }
            orderedLeaves = spatial ? try Self.spatialOrder(points, count: count, device: device) : nil
            self.bounds = bounds
            self.decisions = decisions
            self.densities = densities
            encoder.setComputePipelineState(build)
            encoder.setBuffers([points, harmonics, bounds], offsets: [0, 0, 0], range: 0..<3)
            if let orderedLeaves { encoder.setBuffer(orderedLeaves, offset: 0, index: 4) }
            var first = 0, children = count
            for parents in levels {
                let start = first + children
                var level = SIMD4(UInt32(first), UInt32(children), UInt32(start), UInt32(count))
                encoder.setBytes(&level, length: 16, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: (parents + 127) / 128, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                encoder.memoryBarrier(scope: .buffers)
                first = start
                children = parents
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw error }
        }

        /// Sort once in capture space. The original leaf arrays and equal-depth tie order
        /// stay intact. Companding gives distant outliers less influence on local cell size.
        private static func spatialOrder(_ points: any MTLBuffer, count: Int, device: any MTLDevice) throws -> any MTLBuffer {
            func spread(_ coordinate: Float) -> UInt64 {
                let unit = 0.5 + 0.5 * coordinate / (16 + abs(coordinate))
                var x = UInt64(min(16383, max(0, unit * 16383)))
                x = (x | x << 16) & 0x000000ff0000ff
                x = (x | x << 8) & 0x0000f00f00f00f
                x = (x | x << 4) & 0x000c30c30c30c3
                return (x | x << 2) & 0x00249249249249
            }
            guard let ids = device.makeBuffer(length: count * 8, options: .storageModeShared) else {
                throw RendererError.allocation
            }
            let raw = points.contents()
            var sorted = [UInt64](repeating: 0, count: count)
            for i in 0..<count {
                if i.isMultiple(of: 16384) { try Task.checkCancellation() }
                let p = raw.advanced(by: i * MemoryLayout<Point>.stride)
                let x = spread(p.load(as: Float.self))
                let y = spread(p.load(fromByteOffset: 4, as: Float.self))
                let z = spread(p.load(fromByteOffset: 8, as: Float.self))
                sorted[i] = ((x | (y << 1) | (z << 2)) << 22) | UInt64(i)
            }
            sorted.sort()
            let order = ids.contents().assumingMemoryBound(to: UInt32.self)
            for (slot, key) in sorted.enumerated() {
                let source = UInt32(key & 0x3fffff)
                order[slot] = source
                order[count + Int(source)] = UInt32(slot)
            }
            return ids
        }

        func select(eyes: [Eye], rates: [EyeRate], radius: Float, outerLevels: Bool, finestLevel: Int, on encoder: any MTLComputeCommandEncoder) -> Bool {
            guard finestLevel < 3, (1...2).contains(eyes.count), eyes.count == rates.count, radius > 0 else { return false }
            var views = [View]()
            let threads = MTLSize(width: 128, height: 1, depth: 1)
            for (i, eye) in eyes.enumerated() {
                let rate = rates[i]
                let size = rate.map.screenSize
                let viewport = eye.viewport
                let length = rate.map.parameterDataSizeAndAlign.size
                guard rate.slice >= 0, rate.slice < rate.map.layerCount, length > 0,
                      viewport.originX >= 0, viewport.originY >= 0,
                      viewport.originX + viewport.width <= Double(size.width) + 0.01,
                      viewport.originY + viewport.height <= Double(size.height) + 0.01,
                      let data = device.makeBuffer(length: length, options: .storageModeShared) else { return false }
                // Each submission owns its copy until the consuming command buffer completes.
                rate.map.copyParameterData(buffer: data, offset: 0)
                var info = SIMD4(UInt32(size.width), UInt32(size.height), UInt32(rate.slice), 0)
                encoder.setComputePipelineState(sample)
                encoder.setBuffer(data, offset: 0, index: 0)
                encoder.setBuffer(densities, offset: i * 129 * 8, index: 1)
                encoder.setBytes(&info, length: 16, index: 2)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: threads)
                let focal = SIMD2(abs(eye.projection.columns.0.x) + abs(eye.projection.columns.2.x),
                                  abs(eye.projection.columns.1.y) + abs(eye.projection.columns.2.y))
                views.append(View(clip: eye.projection * eye.view, viewport: SIMD4(Float(viewport.originX),
                    Float(viewport.originY), Float(viewport.width), Float(viewport.height)),
                    mapSize: SIMD2(Float(size.width), Float(size.height)), focal: focal))
            }
            encoder.memoryBarrier(scope: .buffers)
            encoder.setComputePipelineState(classify)
            encoder.setBuffer(bounds, offset: 0, index: 0)
            encoder.setBuffer(decisions, offset: 0, index: 1)
            views.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            encoder.setBuffer(densities, offset: 0, index: 3)
            let limits: [SIMD2<Float>] = outerLevels
                ? [[radius, 0.76], [radius * 2, 0.36], [radius * 4, 0.16]]
                : [[radius, 0.60]]
            var offset = 0
            for (index, entry) in zip(levels, limits).enumerated() {
                let (groups, threshold) = entry
                if index < finestLevel { offset += groups; continue }
                var config = SIMD2(UInt32(groups), UInt32(eyes.count))
                var limit = threshold
                encoder.setBuffer(bounds, offset: offset * 16, index: 0)
                encoder.setBuffer(decisions, offset: 0, index: 1)
                var outputOffset = UInt32(offset)
                encoder.setBytes(&outputOffset, length: 4, index: 6)
                encoder.setBytes(&config, length: 8, index: 4)
                encoder.setBytes(&limit, length: 8, index: 5)
                encoder.dispatchThreadgroups(MTLSize(width: (groups + 127) / 128, height: 1, depth: 1),
                                             threadsPerThreadgroup: threads)
                offset += groups
            }
            encoder.memoryBarrier(scope: .buffers)
            return true
        }
    }
}
