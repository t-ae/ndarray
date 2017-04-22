
import XCTest
import NDArray

class ArrayLiteralTests: XCTestCase {

    func testArrayLiteral1() {
        let a = NDArray([1, 2, 3])
        XCTAssertEqual(a, NDArray(shape: [3], elements: [1, 2, 3]))
    }

    func testArrayLiteral2() {
        let a = NDArray([[0, 1, 2],
                         [3, 4, 5]])
        XCTAssertEqual(a, NDArray(shape: [2, 3], elements: [0, 1, 2, 3, 4, 5]))
    }
    
    func testArrayLiteral3() {
        let a = NDArray([[[0, 1, 2],
                          [3, 4, 5]],
                         [[6, 7, 8],
                          [9, 10, 11]]])
        XCTAssertEqual(a, NDArray(shape: [2, 2, 3],
                                  elements: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    }
}
