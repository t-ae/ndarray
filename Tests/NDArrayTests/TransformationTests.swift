import XCTest
@testable import NDArray

class TransformationTests: XCTestCase {
    
    func testTranspose() {
        let a = NDArray.range(24).reshaped([2, 3, 4]).transposed()
        
        XCTAssertEqual(a,
                       NDArray([[[ 0, 12], [ 4, 16], [ 8, 20]],
                                [[ 1, 13], [ 5, 17], [ 9, 21]],
                                [[ 2, 14], [ 6, 18], [10, 22]],
                                [[ 3, 15], [ 7, 19], [11, 23]]]))
    }
    
    func testMoveAxis() {
        let a = NDArray.range(24).reshaped([2, 3, 4])
        
        XCTAssertEqual(a.moveAxis(from: 0, to: -1),
                       NDArray([[[ 0, 12], [ 1, 13], [ 2, 14], [ 3, 15]],
                                [[ 4, 16], [ 5, 17], [ 6, 18], [ 7, 19]],
                                [[ 8, 20], [ 9, 21], [10, 22], [11, 23]]]))
        
        XCTAssertEqual(a.moveAxis(from: 1, to: 0),
                       NDArray([[[ 0,  1,  2,  3],
                                 [12, 13, 14, 15]],
                                [[ 4,  5,  6,  7],
                                 [16, 17, 18, 19]],
                                [[ 8,  9, 10, 11],
                                 [20, 21, 22, 23]]]))
        
        XCTAssertEqual(a.moveAxis(from: -1, to: 0),
                       NDArray([[[ 0,  4,  8],
                                 [12, 16, 20]],
                                [[ 1,  5,  9],
                                 [13, 17, 21]],
                                [[ 2,  6, 10],
                                 [14, 18, 22]],
                                [[ 3,  7, 11],
                                 [15, 19, 23]]]))
    }
    
    func testSwapAxes() {
        let a = NDArray.range(8).reshaped([2, 2, 2])
        
        XCTAssertEqual(a.swapAxes(0, 1),
                       NDArray([[[0, 1], [4, 5]],
                                [[2, 3], [6, 7]]]))
        XCTAssertEqual(a.swapAxes(1, -1),
                       NDArray([[[0, 2], [1, 3]],
                                [[4, 6], [5, 7]]]))
        XCTAssertEqual(a.swapAxes(0, -1),
                       NDArray([[[0, 4], [2, 6]],
                                [[1, 5], [3, 7]]]))
    }
    
    func testReshape() {
        let a = NDArray.range(10)
        XCTAssertEqual(a.reshaped([-1, 5]),
                       NDArray([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]]))
        XCTAssertEqual(a.reshaped([2, -1]),
                       NDArray([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]]))
    }
    
    func testFlip() {
        do {
            let a = NDArray.range(9).reshaped([3, 3])
            XCTAssertEqual(a.flipped(0),
                           NDArray([[6, 7, 8],
                                    [3, 4, 5],
                                    [0, 1, 2]]))
            
            XCTAssertEqual(a.flipped(1),
                           NDArray([[2, 1, 0],
                                    [5, 4, 3],
                                    [8, 7, 6]]))
            
            XCTAssertEqual(a.flipped(0).flipped(1),
                           NDArray([[8, 7, 6],
                                    [5, 4, 3],
                                    [2, 1, 0]]))
            
            XCTAssertEqual(a.flipped(1).flipped(0),
                           NDArray([[8, 7, 6],
                                    [5, 4, 3],
                                    [2, 1, 0]]))
            
            XCTAssertEqual(a.flipped(0).flipped(0), a)
            XCTAssertEqual(a.flipped(1).flipped(1), a)
        }
        do {
            let a = NDArray(shape: [3, 3],
                            strides: [3, 0],
                            baseOffset: 1,
                            data: [0, 1, 2, 3, 4, 5, 6, 7, 8])
            XCTAssertEqual(a.flipped(0),
                           NDArray([[7, 7, 7],
                                    [4, 4, 4],
                                    [1, 1, 1]]))
            XCTAssertEqual(a.flipped(1),
                           NDArray([[1, 1, 1],
                                    [4, 4, 4],
                                    [7, 7, 7]]))
            XCTAssertEqual(a.flipped(0).flipped(1),
                           NDArray([[7, 7, 7],
                                    [4, 4, 4],
                                    [1, 1, 1]]))
        }
        do {
            let a = NDArray(shape: [3, 3],
                            strides: [0, 1],
                            baseOffset: 3,
                            data: [0, 1, 2, 3, 4, 5, 6, 7, 8])
            XCTAssertEqual(a.flipped(0),
                           NDArray([[3, 4, 5],
                                    [3, 4, 5],
                                    [3, 4, 5]]))
            XCTAssertEqual(a.flipped(1),
                           NDArray([[5, 4, 3],
                                    [5, 4, 3],
                                    [5, 4, 3]]))
            XCTAssertEqual(a.flipped(1),
                           NDArray([[5, 4, 3],
                                    [5, 4, 3],
                                    [5, 4, 3]]))
            XCTAssertEqual(a.flipped(0).flipped(1),
                           NDArray([[5, 4, 3],
                                    [5, 4, 3],
                                    [5, 4, 3]]))
        }
    }
    
}
