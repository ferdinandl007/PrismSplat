import Foundation

/// A binary little-endian Gaussian splat PLY capture.
public struct SplatFile: Sendable {
    public enum Precision: Sendable {
        case float32
        case compact

        func offset(of field: Int) -> Int {
            guard self == .compact else { return field * 4 }
            // Keep centers at full precision and align the half4 rotation to 8 bytes.
            switch field {
            case 0..<3: return field * 4
            case 3..<6: return 24 + (field - 3) * 2
            case 6..<10: return 16 + (field - 6) * 2
            default: return 30 + (field - 10) * 2
            }
        }

        func stride(for fields: Int) -> Int {
            let end = offset(of: fields)
            return self == .compact ? (end + 7) & ~7 : end
        }
    }

    public let count: Int
    public let degree: UInt8
    let usesLinearOpacity: Bool
    let attributes: Data
    let precision: Precision

    var stride: Int { precision.stride(for: 11 + 3 * (Int(degree) + 1) * (Int(degree) + 1)) }

    /// Reads a PLY or an LZFSE-compressed PLY. The limit samples the full vertex range.
    public static func read(from url: URL, limit: Int = 400_000, precision: Precision = .compact) async throws -> Self {
        let reader = Task.detached(priority: .userInitiated) {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let file = try Data(contentsOf: url, options: .alwaysMapped)
            let data = url.pathExtension.lowercased() == "lzfse"
                ? try (file as NSData).decompressed(using: .lzfse) as Data
                : file
            return try decode(data, limit: limit, precision: precision)
        }
        return try await withTaskCancellationHandler {
            try await reader.value
        } onCancel: {
            reader.cancel()
        }
    }

    public static func decode(_ data: Data, limit: Int = 400_000, precision: Precision = .compact) throws -> Self {
        let prefix = data.prefix(65_536)
        guard let end = prefix.range(of: Data("\nend_header\n".utf8)) ??
                        prefix.range(of: Data("\nend_header\r\n".utf8)),
              let header = String(data: prefix[..<end.lowerBound], encoding: .ascii) else {
            throw ImportError.invalidHeader
        }
        let lines = header.split(whereSeparator: \.isNewline)
        guard lines.first == "ply", lines.contains("format binary_little_endian 1.0"),
              let vertex = lines.firstIndex(where: { $0.hasPrefix("element ") }),
              lines[vertex].hasPrefix("element vertex "),
              let count = Int(lines[vertex].dropFirst("element vertex ".count)), count > 0, limit > 0 else {
            throw ImportError.invalidHeader
        }
        let names = try lines.dropFirst(vertex + 1)
            .prefix { !$0.hasPrefix("element ") && !$0.hasPrefix("end_header") }
            .filter { $0.hasPrefix("property ") }
            .map { line in
                let parts = line.split(whereSeparator: \.isWhitespace)
                guard parts.count == 3, parts[1] == "float" || parts[1] == "float32" else {
                    throw ImportError.unsupportedProperties
                }
                return String(parts[2])
            }
        guard names.count <= 128, Set(names).count == names.count else { throw ImportError.unsupportedProperties }
        let columns = Dictionary(uniqueKeysWithValues: names.enumerated().map { ($0.element, $0.offset * 4) })
        let restCount = names.filter { $0.hasPrefix("f_rest_") }.count
        let degrees = [0: UInt8(0), 9: 1, 24: 2, 45: 3]
        guard let degree = degrees[restCount] else { throw ImportError.unsupportedProperties }
        let restPerChannel = restCount / 3
        let usesLinearOpacity = columns["opacity_linear"] != nil
        let opacity = usesLinearOpacity ? "opacity_linear" : "opacity"
        var order = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                     opacity, "f_dc_0", "f_dc_1", "f_dc_2"]
        // PLY groups coefficients by color. Store an RGB triplet for each basis.
        for coefficient in 0..<restPerChannel {
            for channel in 0..<3 { order.append("f_rest_\(channel * restPerChannel + coefficient)") }
        }
        let offsets = order.compactMap { columns[$0] }
        guard offsets.count == order.count else { throw ImportError.unsupportedProperties }
        let vertexStride = names.count * 4
        let vertexStart = end.upperBound - data.startIndex
        guard count <= (data.count - vertexStart) / vertexStride else { throw ImportError.truncated }

        let kept = min(count, limit)
        let recordStride = precision.stride(for: offsets.count)
        var attributes = Data(count: kept * recordStride)
        try attributes.withUnsafeMutableBytes { (output: UnsafeMutableRawBufferPointer) in
            try data.withUnsafeBytes { (input: UnsafeRawBufferPointer) in
                func write(_ value: Float, field: Int, row: Int) {
                    let address = row + precision.offset(of: field)
                    if precision == .compact, field >= 3 {
                        let bound = Float(Float16.greatestFiniteMagnitude)
                        let packed = Float16(min(bound, max(-bound, value)))
                        output.storeBytes(of: packed, toByteOffset: address, as: Float16.self)
                    } else {
                        output.storeBytes(of: value, toByteOffset: address, as: Float.self)
                    }
                }

                for point in 0..<kept {
                    if point.isMultiple(of: 4_096) { try Task.checkCancellation() }
                    let index = point * count / kept
                    let source = vertexStart + index * vertexStride
                    let row = point * recordStride
                    var hasRotation = false
                    for (field, offset) in offsets.enumerated() {
                        let bits = input.loadUnaligned(fromByteOffset: source + offset, as: UInt32.self)
                        let sample = Float(bitPattern: UInt32(littleEndian: bits))
                        var value = sample.isFinite ? sample : 0
                        switch field {
                        case 3...5:
                            // Extreme scales can overflow exp() or produce unbounded overdraw.
                            value = min(6, max(-20, value))
                        case 6...9:
                            hasRotation = hasRotation || abs(value) >= 0.00001
                        case 10 where usesLinearOpacity:
                            // Native splats use standard Gaussian alpha.
                            value = min(1, max(0, value))
                        default: break
                        }
                        write(value, field: field, row: row)
                    }
                    if !hasRotation { write(1, field: 6, row: row) }
                }
            }
        }
        return Self(count: kept, degree: degree, usesLinearOpacity: usesLinearOpacity,
                    attributes: attributes, precision: precision)
    }

    public enum ImportError: LocalizedError {
        case invalidHeader
        case unsupportedProperties
        case truncated

        public var errorDescription: String? {
            switch self {
            case .invalidHeader: "Choose an uncompressed binary Gaussian splat PLY file."
            case .unsupportedProperties: "Export the scene as a Gaussian splat PLY with float32 attributes."
            case .truncated: "The splat file is incomplete. Export or download it again."
            }
        }
    }
}
