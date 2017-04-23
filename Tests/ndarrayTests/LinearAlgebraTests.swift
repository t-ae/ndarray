
import XCTest
import NDArray

class LinearAlgebraTests: XCTestCase {

    func testNorm() {
        do {
            let a = NDArray([1, 2, 3])
            XCTAssertEqual(norm(a), sqrtf(14))
        }
    }
    
    func testInvert() {
        do {
            let a = NDArray([[1, 0], [0, 1]])
            XCTAssertEqual(try! inv(a), a)
            let b = NDArray.stack([a, a, a])
            XCTAssertEqual(try! inv(b), b)
        }
        do {
            let a = NDArray([[1, 2, 3],
                             [1, 3, 5],
                             [2, 4, 5]])
            XCTAssertEqualWithAccuracy(try! inv(a),
                                       NDArray([[ 5, -2, -1],
                                                [-5,  1,  2],
                                                [ 2,  0, -1]]),
                                       accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(8).reshaped([2, 2, 2])
            XCTAssertEqualWithAccuracy(try! inv(a),
                                       NDArray([[[-1.5,  0.5],
                                                 [ 1 ,  0 ]],
                                                [[-3.5,  2.5],
                                                 [ 3 , -2 ]]]),
                                       accuracy: 1e-5)
        }
    }

}
