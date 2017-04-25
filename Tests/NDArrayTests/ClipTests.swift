
import XCTest
import NDArray

class ClipTests: XCTestCase {
    
    func testClipLow() {
        let a = NDArray.range(-3..<6).reshaped([3,3])
        XCTAssertEqual(a.clip(low: 0), NDArray([[0, 0, 0],
                                                [0, 1, 2],
                                                [3, 4, 5]]))
    }
    
    func testClip() {
        let a = NDArray.range(-3..<6).reshaped([3,3])
        XCTAssertEqual(a.clip(low: 0, high: 2), NDArray([[0, 0, 0],
                                                         [0, 1, 2],
                                                         [2, 2, 2]]))
    }

}
