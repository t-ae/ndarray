
import XCTest
import NDArray

class ClipTests: XCTestCase {
    
    func testMax() {
        let a = NDArray.range(-3..<6).reshaped([3,3])
        XCTAssertEqual(max(a, 0), NDArray([[0, 0, 0],
                                           [0, 1, 2],
                                           [3, 4, 5]]))
    }
    
    func testClip() {
        let a = NDArray.range(-3..<6).reshaped([3,3])
        XCTAssertEqual(clip(a, low: 0, high: 2), NDArray([[0, 0, 0],
                                                          [0, 1, 2],
                                                          [2, 2, 2]]))
    }

}
