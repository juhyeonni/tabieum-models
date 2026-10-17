// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "bench",
    platforms: [.macOS(.v15)],
    targets: [.executableTarget(name: "cml")]
)
