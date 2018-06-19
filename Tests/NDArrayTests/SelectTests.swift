import XCTest
import NDArray

class SelectTests: XCTestCase {
    
    func testSelect() {
        let a = NDArray.range(9).reshaped([3, 3])
        XCTAssertEqual(a.select([0, 0, 1]),
                       NDArray([[0, 1, 2],
                                [0, 1, 2],
                                [3, 4, 5]]))
        let noIndices: [Int] = []
        XCTAssertEqual(a.select(noIndices), NDArray.zeros([0, 3]))
        
        XCTAssertEqual(a.select { $0[1].asScalar() < 7 },
                       NDArray([[0, 1, 2], [3, 4, 5]]))
        XCTAssertEqual(a.select { $0[1].asScalar() < 0 },
                       NDArray.zeros([0, 3]))
        
        XCTAssertEqual(a.select([true, false, false]),
                       NDArray([[0, 1, 2]]))
        XCTAssertEqual(a.select([true, false, true]),
                       NDArray([[0, 1, 2],
                                [6, 7, 8]]))
        let noMask: [Bool] = [false, false, false]
        XCTAssertEqual(a.select(noMask), NDArray.zeros([0, 3]))
    }
    
    func testIndices() {
        let a = NDArray.range(9).reshaped([3, 3])
        
        XCTAssertEqual(a.indices { $0[1].asScalar() < 7 }, [0, 1])
    }
}
