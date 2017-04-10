
import XCTest
@testable import NDArray

class TransformationTests: XCTestCase {

    func testTranspose() {
        let a = NDArray.range(0..<24).reshaped([2, 3, 4]).transposed()
        
        XCTAssertEqual(a, NDArray(shape: [4, 3, 2],
                                  elements: [ 0, 12,  4, 16,  8, 20,
                                              1, 13,  5, 17,  9, 21,
                                              2, 14,  6, 18, 10, 22,
                                              3, 15,  7, 19, 11, 23]))
    }
    
    func testReshape() {
        let a = NDArray.range(0..<10)
        XCTAssertEqual(a.reshaped([-1, 5]).shape, [2, 5])
        XCTAssertEqual(a.reshaped([2, -1]).shape, [2, 5])
    }

}
