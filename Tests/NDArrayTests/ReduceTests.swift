
import XCTest
import NDArray

class ReduceTests: XCTestCase {

    func testMax() {
        do {
            let a = NDArray([[3, 2],
                             [1, 0]])
            XCTAssertEqual(max(a, along: 0), NDArray([3, 2]))
            XCTAssertEqual(max(a, along: 0, keepDims: true), NDArray([[3, 2]]))
            XCTAssertEqual(max(a, along: 1), NDArray([3, 1]))
            XCTAssertEqual(max(a, along: 1, keepDims: true), NDArray([[3], [1]]))
        }
    }
    
    func testArgmin() {
        do {
            let a = NDArray([[1, 2, 3],
                             [1, 3, 2],
                             [3, 1, 2]])
            XCTAssertEqual(argmin(a, along: 0), NDArray([0, 2, 1]))
            XCTAssertEqual(argmin(a, along: 1), NDArray([0, 0, 1]))
        }
    }

    func testStd() {
        let a = NDArray.range(9).reshaped([3, 3])
        XCTAssertEqualWithAccuracy(std(a, along: 0),
                                   NDArray([ 2.44948974,  2.44948974,  2.44948974]),
                                   accuracy: 1e-5)
    }
}
