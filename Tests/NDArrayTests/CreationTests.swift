
import XCTest
import NDArray

class CreationTests: XCTestCase {

    func testZeros() {
        let a = NDArray.zeros([2, 2, 2])
        XCTAssertEqual(a, NDArray([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]))
    }
    
    
    func testEye() {
        do {
            let a = NDArray.eye(0)
            XCTAssertEqual(a, NDArray(shape: [0, 0], elements: []))
        }
        do {
            let a = NDArray.eye(1)
            XCTAssertEqual(a, NDArray([[1]]))
        }
        do {
            let a = NDArray.eye(2)
            XCTAssertEqual(a, NDArray([[1, 0], [0, 1]]))
        }
        do {
            let a = NDArray.eye(3)
            XCTAssertEqual(a, NDArray([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]]))
        }
    }
    
    func testDiagonal() {
        do {
            let a = NDArray.diagonal([0, 1, 2, 3])
            XCTAssertEqual(a, NDArray([[0, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 2, 0],
                                       [0, 0, 0, 3]]))
        }
        do {
            let a = NDArray.diagonal(NDArray([[0,1], [2, 3]]))
            XCTAssertEqual(a, NDArray([[[0, 0], [0, 1]],
                                       [[2, 0], [0, 3]]]))
        }
    }
}
