
import XCTest
import NDArray

class LinearAlgebraTests: XCTestCase {

    func testNorm() {
        do {
            let a = NDArray([1, 2, 3])
            XCTAssertEqual(norm(a), sqrtf(14))
        }
    }
    
    func testNorm2() {
        do {
            let a = NDArray.range(9).reshaped([3, 3])
            XCTAssertEqual(norm(a, along: 0), NDArray([6.70820393,  8.1240384 ,  9.64365076]))
            XCTAssertEqual(norm(a, along: 1), NDArray([2.23606798,   7.07106781,  12.20655562]))
        }
    }
    
    func testDeterminant() {
        do {
            let a = NDArray([[1, 2],
                             [3, 4]])
            let ans = try! determinant(a)
            XCTAssertEqual(ans, NDArray(scalar: -2))
        }
        do {
            let a = NDArray([[0, 2, 1],
                             [1, 2, 0],
                             [0, 0, 1]])
            let ans = try! determinant(a)
            XCTAssertEqual(ans, NDArray(scalar: -2))
        }
        do {
            let a = NDArray([[4, 1, 0, 0, 2],
                             [2, 2, 0, 4, 2],
                             [0, 3, 3, 1, 1],
                             [4, 2, 3, 1, 1],
                             [2, 4, 4, 0, 4]])
            let ans = try! determinant(a)
            XCTAssertEqual(ans, NDArray(scalar: 192))
        }
        do {
            let a = NDArray([[[1, 2],
                              [3, 4]],
                             [[1, 2],
                              [2, 1]],
                             [[1, 3],
                              [3, 1]]])
            let ans = try! determinant(a)
            XCTAssertEqual(ans, NDArray([-2, -3, -8]))
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
