
import XCTest
@testable import ndarray

class UtilsTests: XCTestCase {
    
    func testGather() {
        do {
            let a = NDArray.eye(3)
            let elements = gatherElements(a)
            XCTAssertEqual(elements, [1,0,0, 0,1,0, 0,0,1])
        }
        
    }

    func testBroadcast() {
        let a = NDArray(shape: [1, 3, 1], elements: [1, 2, 3])
        let b = NDArray(shape: [1, 3], elements: [1, 2, 3])
        
        let (lhs, rhs) = broadcast(a, b)
        
        XCTAssertEqual(lhs.strides, [3, 1, 0])
        XCTAssertEqual(rhs.strides, [0, 0, 1])
    }
    
    func testA() {
        let a = (0..<1000000).map { $0 }
        measure {
            _ = a.map { $0 }
        }
    }
    
    func testB() {
        let a = (0..<1000000).map { $0 }
        measure {
            let p = UnsafeMutablePointer<Float>.allocate(capacity: a.count)
            defer { p.deallocate(capacity: a.count) }
            memcpy(p, a, a.count*MemoryLayout<Float>.size)
            _ = Array(UnsafeBufferPointer(start: p, count: a.count))
        }
    }
}
