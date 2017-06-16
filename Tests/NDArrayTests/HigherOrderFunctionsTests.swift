
import XCTest
import NDArray

class HigherOrderFunctionsTests: XCTestCase {
    
    func testMapElements() {
        do {
            let a = NDArray([[-1, 0], [1, -2]])
            let ans = a.mapElements { $0 > 0 ? 1 : 0 }
            XCTAssertEqual(ans, NDArray([[0, 0], [1, 0]]))
        }
    }

}
