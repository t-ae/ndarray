
import XCTest
import NDArray

class PowerTests: XCTestCase {

    func testPower() {
        do {
            let a = NDArray([[0, 1],
                             [2, 2]])
            let b = NDArray([2, 3])
            
            XCTAssertEqual(a**b, NDArray([[0, 1],
                                          [4, 8]]))
            
            XCTAssertEqual(a.transposed()**b,
                           NDArray([[0, 8],
                                    [1, 8]]))
        }
        do {
            let a = NDArray([[0, 1],
                             [2, 2]])
            let b = NDArray([2, 3])
            let c = NDArray.zeros([1])
            
            XCTAssertEqual(a**b*c, NDArray([[0, 0],
                                            [0, 0]]))
            XCTAssertEqual(c*a**b, NDArray([[0, 0],
                                            [0, 0]]))
        }
    }
}
