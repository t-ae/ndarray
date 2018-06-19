import XCTest
import NDArray

class CompoundAssignmentTests: XCTestCase {

    func testScalar() {
        do {
            var a = NDArray([[0, 1],
                             [2, 3]])
            a += 1
            XCTAssertEqual(a, NDArray([[1, 2],
                                       [3, 4]]))
        }
        do {
            var a = NDArray([[0, 1],
                             [2, 3]]).transposed()
            a += 1
            XCTAssertEqual(a, NDArray([[1, 3],
                                       [2, 4]]))
        }
    }
    
    func testNDArray() {
        do {
            var a = NDArray([[0, 1],
                             [2, 3]])
            a += a
            XCTAssertEqual(a, NDArray([[0, 2],
                                       [4, 6]]))
        }
        do {
            var a = NDArray([[0, 1],
                             [2, 3]]).transposed()
            a += a.transposed()
            XCTAssertEqual(a, NDArray([[0, 3],
                                       [3, 6]]))
        }
    }
}
