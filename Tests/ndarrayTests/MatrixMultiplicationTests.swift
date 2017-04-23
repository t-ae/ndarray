
import XCTest
import NDArray

class MatrixMultiplicationTests: XCTestCase {

    func testMatmul() {
        do {
            let a = NDArray([[1, 2], [3, 4]])
            let b = NDArray.eye(2)
            let ans = a <*> b
            XCTAssertEqual(ans, a)
        }
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4]).transposed([0, 2, 1])
            let b = NDArray.eye(3)
            let ans = a <*> b
            XCTAssertEqual(ans, a)
        }
    }

}
