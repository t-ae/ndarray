
import XCTest
import NDArray

class MatrixMultiplicationTests: XCTestCase {

    func testMatmul() {
        do {
            let a = NDArray([[1, 2], [3, 4]])
            let b = NDArray.eye(2)
            let c = a |*| b
            let d = b |*| a
            XCTAssertEqual(c, a)
            XCTAssertEqual(d, a)
        }
        do {
            let a = NDArray.range(9).reshaped([3, 3])
            let b = a[1..<2]
            let c = a[nil, 1..<2]
            XCTAssertEqual(b |*| c, NDArray(shape: [1, 1], elements: [54]))
            XCTAssertEqual(c |*| b, NDArray([[ 3,  4,  5],
                                             [12, 16, 20],
                                             [21, 28, 35]]))
        }
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4]).transposed([0, 2, 1])
            let b = NDArray.eye(3)
            let c = a |*| b
            XCTAssertEqual(c, a)
        }
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4]).transposed()
            let b = NDArray([[1, 2], [3, 4]]).transposed()
            let ans = a |*| b
            XCTAssertEqual(ans,
                           NDArray([[[ 24,  48],
                                     [ 36,  76],
                                     [ 48, 104]],
                                    
                                    [[ 27,  55],
                                     [ 39,  83],
                                     [ 51, 111]],
                                    
                                    [[ 30,  62],
                                     [ 42,  90],
                                     [ 54, 118]],
                                    
                                    [[ 33,  69],
                                     [ 45,  97],
                                     [ 57, 125]]]))
            
            let ans2 = a[0] |*| b
            XCTAssertEqual(ans2,
                           NDArray([[ 24,  48],
                                    [ 36,  76],
                                    [ 48, 104]]))
        }
        do {
            let a = NDArray.eye(3)
            let b = NDArray.ones([3, 3, 3])
            let ans = a |*| b
            XCTAssertEqual(ans, b)
        }
        do {
            let a = NDArray([1, 2, 3]).expandDims(1)
            let b = NDArray([1, 2, 3]).expandDims(0)
            let ans = a |*| b
            XCTAssertEqual(ans, NDArray([[1, 2, 3],
                                         [2, 4, 6],
                                         [3, 6, 9]]))
        }
    }

}
