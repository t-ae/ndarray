
import XCTest
import NDArray

class LinearAlgebraTests: XCTestCase {

    func testNorm() {
        do {
            let a = NDArray([1, 2, 3])
            XCTAssertEqual(norm(a), sqrtf(14))
        }
    }

}
