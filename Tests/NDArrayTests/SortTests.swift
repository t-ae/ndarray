import XCTest
import NDArray
import Accelerate

class SortTests: XCTestCase {

    func testSort() {
        do {
            let a = NDArray([0, -1, 2, -3, 4, -5, 6, -7, 0.1, -0.2])
            XCTAssertEqual(sort(a, ascending: true), NDArray([-7, -5, -3, -1, -0.2, 0, 0.1, 2, 4, 6]))
            XCTAssertEqual(sort(a, ascending: false), NDArray([6, 4, 2, 0.1, 0, -0.2, -1, -3, -5, -7]))
        }
        do {
            let a = NDArray([[ 1,  0,  2],
                             [-1,  0, -2],
                             [ 5,  4,  0]])
            XCTAssertEqual(sort(a, along: 1), NDArray([[ 0,  1,  2],
                                                       [-2, -1,  0],
                                                       [ 0,  4,  5]]))
            XCTAssertEqual(sort(a, along: 0), NDArray([[-1,  0, -2],
                                                       [ 1,  0,  0],
                                                       [ 5,  4,  2]]))
        }
    }
    
    
    func testArgsort() {
        let a = NDArray([0, -1, 2, -3, 4, -5, 6, -7, 0.1, -0.2])
        XCTAssertEqual(argsort(a, ascending: true), [7, 5, 3, 1, 9, 0, 8, 2, 4, 6])
        XCTAssertEqual(argsort(a, ascending: false), [6, 4, 2, 8, 0, 9, 1, 3, 5, 7])
    }
 
}
