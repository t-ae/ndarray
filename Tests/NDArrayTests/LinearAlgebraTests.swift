
import XCTest
import NDArray

class LinearAlgebraTests: XCTestCase {

    func testVectorNorm() {
        do {
            let a = NDArray([1, 2, 3])
            XCTAssertEqual(vectorNorm(a), NDArray(scalar: sqrtf(14)))
        }
        do {
            let a = NDArray.range(9).reshaped([3, 3])
            XCTAssertEqual(vectorNorm(a, axis: 0), NDArray([6.70820393,  8.1240384 ,  9.64365076]))
            XCTAssertEqual(vectorNorm(a, axis: 1), NDArray([2.23606798,   7.07106781,  12.20655562]))
            
            XCTAssertEqual(vectorNorm(a, axis: 0, keepDims: true),
                           NDArray([[6.70820393,  8.1240384 ,  9.64365076]]))
            XCTAssertEqual(vectorNorm(a, axis: 1, keepDims: true),
                           NDArray([[2.23606798],  [7.07106781], [12.20655562]]))
        }
    }
    
    func testMatrixNorm() {
        do {
            let a = NDArray.range(9).reshaped([3, 3])
            XCTAssertEqual(matrixNorm(a), NDArray(scalar: 14.282856857085701))
        }
        do {
            let a = NDArray.range(27).reshaped([3, 3, 3])
            print(a.ndim)
            XCTAssertEqual(matrixNorm(a, axes: (1, 2)), NDArray([ 14.28285686,  39.7617907 ,  66.4529909 ]))
            XCTAssertEqual(matrixNorm(a, axes: (0, 2)), NDArray([ 37.30951621,  44.86646855,  52.87721627]))
            XCTAssertEqual(matrixNorm(a, axes: (0, 1)), NDArray([ 42.84857057,  45.39823785,  48.0       ]))
            
            XCTAssertEqual(matrixNorm(a, axes: (1, 2), keepDims: true),
                           NDArray([ [[14.28285686]],  [[39.7617907]] ,  [[66.4529909]] ]))
            XCTAssertEqual(matrixNorm(a, axes: (0, 2), keepDims: true),
                           NDArray([ [[37.30951621],  [44.86646855],  [52.87721627]]]))
            XCTAssertEqual(matrixNorm(a, axes: (0, 1), keepDims: true),
                           NDArray([[[ 42.84857057,  45.39823785,  48.0       ]]]))
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
    
    func testSVD() {
        do {
            let a = NDArray.range(1..<9).reshaped([4, 2])
            let (u, s, vt) = try! svd(a)
            var S = NDArray.zeros([4, 2])
            S[0..<2, 0..<2] = NDArray.diagonal(s)
            let ans = u |*| S |*| vt
            XCTAssertEqualWithAccuracy(ans, a, accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(12).reshaped([4, 3])
            let (u, s, vt) = try! svd(a)
            var S = NDArray.zeros([4, 3])
            S[0..<3, 0..<3] = NDArray.diagonal(s)
            let ans = u |*| S |*| vt
            XCTAssertEqualWithAccuracy(ans, a, accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(3*4*5).reshaped([3, 4, 5])
            let (u, s, vt) = try! svd(a)
            var S = NDArray.zeros([3, 4, 5])
            S[nil, 0..<4, 0..<4] = NDArray.diagonal(s)
            let ans = u |*| S |*| vt
            XCTAssertEqualWithAccuracy(ans, a, accuracy: 1e-3)
        }
        do {
            let a = NDArray.range(1..<9).reshaped([4, 2])
            let (u, s, vt) = try! svd(a, fullMatrices: false)
            let ans = u |*| (s.expandDims(-1) * vt)
            XCTAssertEqualWithAccuracy(ans, a, accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(1..<9).reshaped([2, 4])
            let (u, s, vt) = try! svd(a, fullMatrices: false)
            let ans = u |*| (s.expandDims(-1) * vt)
            XCTAssertEqualWithAccuracy(ans, a, accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(12).reshaped([3, 4])
            let (u, s, vt) = try! svd(a, fullMatrices: false)
            let ans = u |*| (s.expandDims(-1) * vt)
            XCTAssertEqualWithAccuracy(ans, a, accuracy: 1e-5)
        }
    }
    
    func testPinv() {
        do {
            let a = NDArray([[1, 0],
                             [0, 1]])
            XCTAssertEqualWithAccuracy(try! pinv(a), a, accuracy: 1e-3)
        }
        do {
            let a = NDArray([[1, 2, 3],
                             [1, 3, 5],
                             [2, 4, 5]])
            XCTAssertEqualWithAccuracy(try! pinv(a),
                                       NDArray([[ 5, -2, -1],
                                                [-5,  1,  2],
                                                [ 2,  0, -1]]),
                                       accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(3*4).reshaped([3, 4])
            let apinv = try! pinv(a)
            XCTAssertEqualWithAccuracy(a,
                                       a |*| apinv |*| a,
                                       accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(3*4).reshaped([4, 3])
            let apinv = try! pinv(a)
            XCTAssertEqualWithAccuracy(a,
                                       a |*| apinv |*| a,
                                       accuracy: 1e-5)
        }
        do {
            let a = NDArray.range(10*20).reshaped([10, 20])
            let apinv = try! pinv(a)
            XCTAssertEqualWithAccuracy(a,
                                       a |*| apinv |*| a,
                                       accuracy: 1e-3)
        }
    }
    
    func testMatrixRank() {
        do {
            let a = NDArray.eye(5)
            let rank = matrixRank(a)
            XCTAssertEqual(rank, 5)
        }
        do {
            let a = NDArray([[0, 1, 2, 3, 4],
                             [1, 2, 3, 4, 5],
                             [0, 2, 4, 6, 8]])
            let rank = matrixRank(a)
            XCTAssertEqual(rank, 2)
        }
        do {
            let a = NDArray.range(100).reshaped([2, 50])
            let rank = matrixRank(a)
            XCTAssertEqual(rank, 2)
        }
        do {
            let a = NDArray.range(100).reshaped([50, 2])
            let rank = matrixRank(a)
            XCTAssertEqual(rank, 2)
        }
        do {
            let a = NDArray([[ 0,  1,  2,  3,  4,  5],
                             [ 1,  3,  5,  6,  7,  8],
                             [ 0,  2,  4,  5,  6,  7],
                             [-1, -2, -3, -4, -5, -6],
                             [ 1,  4,  7,  9, 11, 13]])
            let rank = matrixRank(a)
            XCTAssertEqual(rank, 3)
        }
    }

}
