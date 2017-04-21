
import XCTest
@testable import NDArray

class UtilsTests: XCTestCase {
    
    func testIsDense() {
        do {
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [12, 4, 1]))
        }
        do {
            // permute
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [12, 1, 4]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [1, 4, 12]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [1, 12, 4]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [4, 1, 12]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [4, 12, 1]))
        }
        do {
            // contain 0
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [0, 4, 1]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [12, 4, 0]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [12, 0, 1]))
            
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [1, 0, 4]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [1, 12, 0]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [0, 12, 4]))
            
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [0, 0, 4]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [1, 0, 0]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [0, 12, 0]))
        }
    }
    
    func testStridedDims() {
        do {
            let strDims = stridedDims(shape: [2, 2, 2], strides: [4, 2, 1])
            XCTAssertEqual(strDims, 3)
        }
        do {
            let strDims = stridedDims(shape: [2, 2], strides: [4, 2])
            XCTAssertEqual(strDims, 2)
        }
        do {
            let strDims = stridedDims(shape: [2, 2], strides: [2, 4])
            XCTAssertEqual(strDims, 1)
        }
        do {
            let strDims = stridedDims(shape: [2, 2], strides: [0, 0])
            XCTAssertEqual(strDims, 2)
        }
        do {
            let strDims = stridedDims(shape: [2, 1, 2, 2], strides: [4, 0, 2, 1])
            XCTAssertEqual(strDims, 4)
        }
    }
    
    func testGather() {
        do {
            let a = NDArray.eye(3)
            let elements = gatherElements(a)
            XCTAssertEqual(elements, [1,0,0, 0,1,0, 0,0,1])
        }
        do {
            let a = NDArray(shape: [0, 2, 3], elements: [])
            let elements = gatherElements(a)
            XCTAssertEqual(elements, [])
            let elements2 = gatherElements(a, forceUniqueReference: true)
            XCTAssertEqual(elements2, [])
        }
        do {
            let a = NDArray(shape: [0, 2, 3], elements: []).transposed()
            let elements = gatherElements(a)
            XCTAssertEqual(elements, [])
        }
    }

    func testBroadcast() {
        let a = NDArray(shape: [1, 3, 1], elements: [1, 2, 3])
        let b = NDArray(shape: [1, 3], elements: [1, 2, 3])
        
        let (lhs, rhs) = broadcast(a, b)
        
        XCTAssertEqual(lhs.strides, [3, 1, 0])
        XCTAssertEqual(rhs.strides, [0, 0, 1])
    }
}
