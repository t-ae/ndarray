import XCTest
import NDArray

class CopySignTests: XCTestCase {
    
    func testCopySign() {
        do {
            let a = NDArray([0, -1, -1, 2])
            let b = NDArray([1, -3, 2, -5])
            
            XCTAssertEqual(copySign(magnitude: a, sign: b), NDArray([0, -1, 1, -2]))
            XCTAssertEqual(copySign(magnitude: b, sign: a), NDArray([1, -3, -2, 5]))
        }
        do {
            XCTAssertEqual(1/copySign(magnitude: NDArray(scalar: 0), sign: NDArray(scalar: 1)),
                                      NDArray(scalar: Float.infinity))
            XCTAssertEqual(1/copySign(magnitude: NDArray(scalar: 0), sign: NDArray(scalar: -1)),
                                      NDArray(scalar: -Float.infinity))
        }
    }
}
