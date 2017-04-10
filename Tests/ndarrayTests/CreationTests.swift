
import XCTest
@testable import NDArray

class CreationTests: XCTestCase {

    func testZeros() {
        let a = NDArray.zeros([2, 2, 2])
        XCTAssertEqual(a, NDArray(shape: [2, 2, 2], elements: [0, 0, 0, 0, 0, 0, 0, 0]))
    }
    
    func testEye() {
        do {
            let a = NDArray.eye(1)
            XCTAssertEqual(a, NDArray(shape: [1, 1], elements: [1]))
        }
        do {
            let a = NDArray.eye(2)
            XCTAssertEqual(a, NDArray(shape: [2, 2], elements: [1, 0, 0, 1]))
        }
        do {
            let a = NDArray.eye(3)
            XCTAssertEqual(a, NDArray(shape: [3, 3], elements: [1, 0, 0, 0, 1, 0, 0, 0, 1]))
        }
    }
}
