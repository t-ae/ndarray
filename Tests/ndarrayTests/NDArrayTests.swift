import XCTest
@testable import NDArray
import Accelerate

class NDArrayTests: XCTestCase {
    func testIsContinuous() {
        do {
            // continuous
            let a = NDArray.range(0..<8).reshaped([2, 2, 2])
            XCTAssertEqual(a.isContinuous, true)
            
            XCTAssertEqual(a[0].isContinuous, true)
            XCTAssertEqual(a[1].isContinuous, true)
            
            XCTAssertEqual(a[0, 0].isContinuous, true)
            XCTAssertEqual(a[1, 1].isContinuous, true)
            
            XCTAssertEqual(a[0, 0, 0].isContinuous, true)
            XCTAssertEqual(a[1, 1, 1].isContinuous, true)
            
            XCTAssertEqual(a[nil, 0].isContinuous, false)
            XCTAssertEqual(a[nil, 1].isContinuous, false)
            
            XCTAssertEqual(a[nil, 0, 0].isContinuous, false)
            XCTAssertEqual(a[nil, 1, 1].isContinuous, false)
            
            XCTAssertEqual(a[nil, nil, 1].isContinuous, false)
        }
        do {
            // uncontinuous
            let a = NDArray.range(0..<8).reshaped([2, 2, 2]).transposed()
            XCTAssertEqual(a.isContinuous, false)
            
            XCTAssertEqual(a[0].isContinuous, false)
            XCTAssertEqual(a[1].isContinuous, false)
            
            XCTAssertEqual(a[0, 0].isContinuous, false)
            XCTAssertEqual(a[1, 1].isContinuous, false)
            
            XCTAssertEqual(a[0, 0, 0].isContinuous, true)
            XCTAssertEqual(a[1, 1, 1].isContinuous, true)
            
            XCTAssertEqual(a[nil, 0].isContinuous, false)
            XCTAssertEqual(a[nil, 1].isContinuous, false)
            
            XCTAssertEqual(a[nil, 0, 0].isContinuous, true)
            XCTAssertEqual(a[nil, 1, 1].isContinuous, true)
            
            XCTAssertEqual(a[nil, nil, 1].isContinuous, false)
        }
    }
    
    static var allTests = [
        ("testIsContinuous", testIsContinuous),
    ]
}
