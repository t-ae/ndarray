
import XCTest
import NDArray

class MatrixMultiplicationTests: XCTestCase {

    func testMatmul() {
        do {
            let a = NDArray([[1, 2], [3, 4]])
            let b = NDArray.eye(2)
            let ans = a <*> b
            XCTAssertEqual(ans, a)
        }
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4]).transposed([0, 2, 1])
            let b = NDArray.eye(3)
            let ans = a <*> b
            XCTAssertEqual(ans, a)
        }
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4]).transposed()
            let b = NDArray([[1, 2], [3, 4]]).transposed()
            let ans = a <*> b
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
        }
    }

}
