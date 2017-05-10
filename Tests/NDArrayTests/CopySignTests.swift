
import XCTest
import NDArray

class CopySignTests: XCTestCase {
    
    func testCipySign() {
        let a = NDArray([0, -1, -1, 2])
        let b = NDArray([1, -3, 2, -5])
        
        XCTAssertEqual(copySign(magnitude: a, sign: b), NDArray([0, -1, 1, -2]))
        XCTAssertEqual(copySign(magnitude: b, sign: a), NDArray([1, -3, -2, 5]))
    }
    
}
