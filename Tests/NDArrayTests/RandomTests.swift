
import XCTest
import NDArray

class RandomTests: XCTestCase {
    
    func testUniform() {
        do {
            let a = NDArray.uniform(shape: [10_000_000])
            XCTAssertEqual(mean(a).asScalar(), 0.5, accuracy: 1e-3)
        }
        do {
            let a = NDArray.uniform(low: -3, high: 3, shape: [10_000_000])
            XCTAssertEqual(mean(a).asScalar(), 0, accuracy: 1e-3)
        }
    }
    
    func testNormal() {
        do {
            let a = NDArray.normal(shape: [10_000_000])
            XCTAssertEqual(mean(a).asScalar(), 0, accuracy: 1e-3)
            XCTAssertEqual(stddev(a).asScalar(), 1, accuracy: 1e-3)
        }
        do {
            let a = NDArray.normal(mu: -1, sigma: 0.5, shape: [10_000_000])
            XCTAssertEqual(mean(a).asScalar(), -1, accuracy: 1e-3)
            XCTAssertEqual(stddev(a).asScalar(), 0.5, accuracy: 1e-3)
        }
    }
}
