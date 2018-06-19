import XCTest
import NDArray

class ClipTests: XCTestCase {
    
    func testMinimum() {
        do {
            let a = NDArray([[1, 2, 3],
                             [4, 5, 6]])
            let b = NDArray([[2],
                             [5]])
            XCTAssertEqual(minimum(a, b), NDArray([[1, 2, 2],
                                                   [4, 5, 5]]))
        }
        do {
            let a = NDArray([[1, 2, 3],
                             [4, 5, 6]]).transposed()
            let b = NDArray([2, 5])
            XCTAssertEqual(minimum(a, b), NDArray([[1, 4],
                                                   [2, 5],
                                                   [2, 5]]))
        }
    }
    
    func testClippedLow() {
        let a = NDArray.range(-3..<6).reshaped([3,3])
        XCTAssertEqual(a.clipped(low: 0), NDArray([[0, 0, 0],
                                                   [0, 1, 2],
                                                   [3, 4, 5]]))
    }
    
    func testClipped() {
        let a = NDArray.range(-3..<6).reshaped([3,3])
        XCTAssertEqual(a.clipped(low: 0, high: 2), NDArray([[0, 0, 0],
                                                            [0, 1, 2],
                                                            [2, 2, 2]]))
    }
    
}
