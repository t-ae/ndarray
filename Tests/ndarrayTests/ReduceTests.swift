
import XCTest
@testable import NDArray

class ReduceTests: XCTestCase {

    func testMax() {
        do {
            let a = NDArray([[3, 2],
                             [1, 0]])
            XCTAssertEqual(max(a, along: 0), NDArray([3, 2]))
            XCTAssertEqual(max(a, along: 1), NDArray([3, 1]))
        }
    }

}
