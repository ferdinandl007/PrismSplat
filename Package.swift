// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PrismSplat",
    platforms: [.macOS(.v15), .iOS(.v18), .visionOS(.v2)],
    products: [.library(name: "PrismSplat", targets: ["PrismSplat"])],
    targets: [
        .target(name: "PrismSplat", resources: [.copy("Shaders")]),
        .testTarget(name: "PrismSplatTests", dependencies: ["PrismSplat"]),
    ]
)
