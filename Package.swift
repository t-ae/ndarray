// swift-tools-version:3.1

import PackageDescription

let package = Package(
    name: "NDArray",
    dependencies: [
        .Package(url: "https://github.com/t-ae/xorswift.git", Version("0.0.9"))
    ]
)
