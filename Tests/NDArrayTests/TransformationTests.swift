
import XCTest
import NDArray

class TransformationTests: XCTestCase {
    
    func testTranspose() {
        let a = NDArray.range(0..<24).reshaped([2, 3, 4]).transposed()
        
        XCTAssertEqual(a,
                       NDArray([[[ 0, 12], [ 4, 16], [ 8, 20]],
                                [[ 1, 13], [ 5, 17], [ 9, 21]],
                                [[ 2, 14], [ 6, 18], [10, 22]],
                                [[ 3, 15], [ 7, 19], [11, 23]]]))
    }
    
    func testReshape() {
        let a = NDArray.range(0..<10)
        XCTAssertEqual(a.reshaped([-1, 5]).shape, [2, 5])
        XCTAssertEqual(a.reshaped([2, -1]).shape, [2, 5])
    }
    
    func testFlip() {
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
}
