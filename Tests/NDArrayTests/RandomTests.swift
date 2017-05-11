
import XCTest
import NDArray

class RandomTests: XCTestCase {
    
    func testUniform() {
        let a = NDArray.uniform(shape: [10_000_000])
        XCTAssertEqualWithAccuracy(mean(a).asScalar(), 0.5, accuracy: 1e-3)
    }
    
    func testNormal() {
        let a = NDArray.normal(shape: [10_000_000])
        XCTAssertEqualWithAccuracy(mean(a).asScalar(), 0, accuracy: 1e-3)
        XCTAssertEqualWithAccuracy(std(a).asScalar(), 1, accuracy: 1e-3)
    }
}
