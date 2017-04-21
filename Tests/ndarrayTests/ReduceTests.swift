
import XCTest
@testable import NDArray

class ReduceTests: XCTestCase {

    func testMax() {
        do {
            let a = NDArray(shape: [2, 2], elements: [3, 2, 1, 0])
            XCTAssertEqual(max(a, along: 0), NDArray(shape: [2], elements: [3, 2]))
            XCTAssertEqual(max(a, along: 1), NDArray(shape: [2], elements: [3, 1]))
        }
    }

}
