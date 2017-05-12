
import XCTest
@testable import NDArray

class UtilsTests: XCTestCase {
    
    func testIsDense() {
        do {
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [12, 4, 1]))
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [6, 4, 1]))
        }
        do {
            // permute
            XCTAssertTrue(isDense(shape: [2, 4, 3], strides: [12, 1, 4]))
            XCTAssertTrue(isDense(shape: [4, 3, 2], strides: [1, 4, 12]))
            XCTAssertTrue(isDense(shape: [4, 2, 3], strides: [1, 12, 4]))
            XCTAssertTrue(isDense(shape: [3, 4, 2], strides: [4, 1, 12]))
            XCTAssertTrue(isDense(shape: [3, 2, 4], strides: [4, 12, 1]))
        }
        do {
            // contain 0
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [0, 4, 1]))
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [12, 4, 0]))
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [12, 0, 1]))
            
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [1, 0, 4]))
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [1, 12, 0]))
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [0, 12, 4]))
            
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [0, 0, 4]))
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [1, 0, 0]))
            XCTAssertFalse(isDense(shape: [2, 3, 4], strides: [0, 12, 0]))
            
            XCTAssertTrue(isDense(shape: [2, 3, 4], strides: [0, 0, 0]))
        }
        do {
            // one
            XCTAssertTrue(isDense(shape: [1], strides: [1]))
        }
    }
    
    func testGetStridedDims() {
        do {
            let strDims = getStridedDims(shape: [2, 2, 2], strides: [4, 2, 1])
            XCTAssertEqual(strDims, 3)
        }
        do {
            let strDims = getStridedDims(shape: [2, 2], strides: [4, 2])
            XCTAssertEqual(strDims, 2)
        }
        do {
            let strDims = getStridedDims(shape: [2, 2], strides: [2, 4])
            XCTAssertEqual(strDims, 1)
        }
        do {
            let strDims = getStridedDims(shape: [2, 2], strides: [0, 0])
            XCTAssertEqual(strDims, 2)
        }
        do {
            let strDims = getStridedDims(shape: [2, 1, 2, 2], strides: [4, 0, 2, 1])
            XCTAssertEqual(strDims, 4)
        }
        do {
            let strDims = getStridedDims(shape: [3, 3], strides: [-3, -1])
            XCTAssertEqual(strDims, 2)
        }
        do {
            let strDims = getStridedDims(shape: [3, 3], strides: [3, -1])
            XCTAssertEqual(strDims, 1)
        }
        do {
            let strDims = getStridedDims(shape: [3, 3], strides: [-6, -2])
            XCTAssertEqual(strDims, 2)
        }
    }
    
    func testGatherElements() {
        do {
            let a = NDArray.eye(3)
            let elements = gatherElements(a)
            XCTAssertEqual(elements.asArray(), [1,0,0, 0,1,0, 0,0,1])
        }
        do {
            let a = NDArray(shape: [0, 2, 3], elements: [])
            let elements = gatherElements(a)
            XCTAssertEqual(elements.asArray(), [])
            let elements2 = gatherElements(a)
            XCTAssertEqual(elements2.asArray(), [])
        }
        do {
            let a = NDArray(shape: [0, 2, 3], elements: []).transposed()
            let elements = gatherElements(a)
            XCTAssertEqual(elements.asArray(), [])
        }
        do {
            let a = NDArray(shape: [3, 3],
                            strides: [3, -1],
                            baseOffset: 2,
                            data: NDArrayData([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
            let elements = gatherElements(a)
            XCTAssertEqual(elements.asArray(), [2, 1, 0, 5, 4, 3, 8, 7, 6])
        }
        do {
            let a = NDArray(shape: [2, 2],
                            strides: [4, -2],
                            baseOffset: 2,
                            data: NDArrayData([2, -1, 1, -1, 4, -1, 3, -1]))
            let elements = gatherElements(a)
            XCTAssertEqual(elements.asArray(), [1, 2, 3, 4])
        }
        do {
            let a = NDArray([[0, 1], [2, 3]])
            XCTAssertEqual(a.flipped(0), NDArray([[2, 3], [0, 1]]))
            XCTAssertEqual(a.flipped(1), NDArray([[1, 0], [3, 2]]))
            XCTAssertEqual(a.flipped(1).flipped(0), NDArray([[3, 2], [1, 0]]))
            XCTAssertEqual(a.flipped(0).flipped(1), NDArray([[3, 2], [1, 0]]))
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
