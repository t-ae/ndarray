
import XCTest
import NDArray

class ArithmeticTests: XCTestCase {
    
    func testNeg() {
        do {
            // contiguous
            let a = NDArray.range(8).reshaped([2, 2, 2])[1]
            let b = -a
            XCTAssertEqual(b, NDArray([[-4, -5],
                                       [-6, -7]]))
        }
        do {
            // dense, uncontiguous
            let a = NDArray.range(8).reshaped([2, 2, 2]).transposed()
            let b = -a
            XCTAssertEqual(b, NDArray([[[-0, -4],
                                        [-2, -6]],
                                       [[-1, -5],
                                        [-3, -7]]]))
        }
        do {
            // not dense
            let a = NDArray.range(8).reshaped([2, 2, 2])[nil, 1]
            let b = -a
            XCTAssertEqual(b, NDArray([[-2, -3],
                                       [-6, -7]]))
        }
        do {
            // flipped
            let a = NDArray.range(4).reshaped([2, 2])
            XCTAssertEqual(-a.flipped(0), NDArray([[-2, -3],
                                                   [ 0, -1]]))
            XCTAssertEqual(-a.flipped(1), NDArray([[-1,  0],
                                                   [-3, -2]]))
        }
    }

    func testAdd() {
        do {
            // scalar + scalar
            let a: NDArray = NDArray(scalar: 1)
            let b: NDArray = NDArray(scalar: 2)
            let c = a + b
            XCTAssertEqual(c.asScalar(), 3)
        }
        do {
            // scalar + NDArray
            let a: NDArray = NDArray(scalar: 1)
            let b = NDArray.range(24).reshaped([2, 3, 4])
            let c = a + b
            XCTAssertEqual(c, NDArray([[[ 1,  2,  3,  4],
                                        [ 5,  6,  7,  8],
                                        [ 9, 10, 11, 12]],
                                       [[13, 14, 15, 16],
                                        [17, 18, 19, 20],
                                        [21, 22, 23, 24]]]))
            let d = b + a
            XCTAssertEqual(d, NDArray([[[ 1,  2,  3,  4],
                                        [ 5,  6,  7,  8],
                                        [ 9, 10, 11, 12]],
                                       [[13, 14, 15, 16],
                                        [17, 18, 19, 20],
                                        [21, 22, 23, 24]]]))
        }
        do {
            // broadcast
            let a = NDArray.range(8).reshaped([2, 2, 2])
            let b = NDArray.range(4).reshaped([2, 2])
            
            XCTAssertEqual(a+b, NDArray([[[0,  2],
                                          [4,  6]],
                                         [[4,  6],
                                          [8, 10]]]))
            XCTAssertEqual(b+a, NDArray([[[0,  2],
                                          [4,  6]],
                                         [[4,  6],
                                          [8, 10]]]))
            XCTAssertEqual(a[1]+b, NDArray([[4,  6],
                                            [8, 10]]))
            XCTAssertEqual(b+a[1], NDArray([[4,  6],
                                            [8, 10]]))
            
            // permuted
            let c = a.transposed()
            XCTAssertEqual(b+c, NDArray([[[ 0,  5],
                                          [ 4,  9]],
                                         [[ 1,  6],
                                          [ 5, 10]]]))
            XCTAssertEqual(c+b, NDArray([[[ 0,  5],
                                          [ 4,  9]],
                                         [[ 1,  6],
                                          [ 5, 10]]]))
            XCTAssertEqual(b+c[1], NDArray([[1, 6],
                                            [5, 10]]))
            XCTAssertEqual(c[1]+b, NDArray([[1, 6],
                                            [5, 10]]))
            
            // both permuted
            let d = b.transposed()
            XCTAssertEqual(c+d, NDArray([[[ 0,  6],
                                          [ 3,  9]],
                                         [[ 1,  7],
                                          [ 4, 10]]]))
            XCTAssertEqual(d+c, NDArray([[[ 0,  6],
                                          [ 3,  9]],
                                         [[ 1,  7],
                                          [ 4, 10]]]))
            XCTAssertEqual(c[1]+d, NDArray([[ 1,  7],
                                            [ 4, 10]]))
            XCTAssertEqual(d+c[1], NDArray([[ 1,  7],
                                            [ 4, 10]]))
        }
        do {
            // empty array
            let a = NDArray(shape: [0, 2, 2], elements: [])
            let b = NDArray([1, 2])
            
            XCTAssertEqual(a+b, NDArray(shape: [0, 2, 2], elements: []))
            XCTAssertEqual(b+a, NDArray(shape: [0, 2, 2], elements: []))
        }
    }
    
    func testSubtract() {
        do {
            let a = NDArray([[0, 1], [2, 3]])
            let b: Float = 2
            XCTAssertEqual(a - b, NDArray([[-2, -1],
                                           [0, 1]]))
            XCTAssertEqual(b - a, NDArray([[2, 1],
                                           [0, -1]]))
        }
        do {
            let a = NDArray([[0, 1], [2, 3]])
            let b = NDArray([1, 4])
            
            XCTAssertEqual(a-b, NDArray([[-1, -3],
                                         [1, -1]]))
        }
    }
    
    func testMultiply() {
        do {
            let a = NDArray([[0, 1], [2, 3]])
            let b: Float = 2
            XCTAssertEqual(a*b, NDArray([[0, 2],
                                         [4, 6]]))
            XCTAssertEqual(b*a, NDArray([[0, 2],
                                         [4, 6]]))
        }
        do {
            let a = NDArray([[0, 1], [2, 3]])
            let b = NDArray([1, 4])
            
            XCTAssertEqual(a*b, NDArray([[0, 4],
                                         [2, 12]]))
        }
    }
    
    func testDivide() {
        do {
            // scalar
            let a = NDArray([[1, 2], [4, 8]])
            let b: Float = 4
            XCTAssertEqual(a / b, NDArray([[0.25, 0.5],
                                           [1,    2]]))
            XCTAssertEqual(b / a,
                                       NDArray([[4, 2],
                                                [1, 0.5]]),
                                       accuracy: 1e-5)
        }
        do {
            let a = NDArray([[0, 1], [2, 3]])
            let b = NDArray([1, 4])
            
            let c: [[Float]] = [[0, 1.0/4],
                                [2, 3.0/4]]
            
            XCTAssertEqual(a/b, NDArray(c), accuracy: 1e-5)
        }
    }
}
