
import XCTest
@testable import NDArray

class NDArrayDataTests: XCTestCase {

    func testCoW() {
        do {
            let data = NDArrayData(value: 0, count: 5)
            var b = data
            b[0] = 100
            XCTAssertEqual(data.asArray(), [0, 0, 0, 0, 0])
            XCTAssertEqual(b.asArray(), [100, 0, 0, 0, 0])
        }
        do {
            let data = NDArrayData(value: 0, count: 5)
            var b = data
            b.withUnsafeMutablePointer { buf in
                buf.pointee = 100
            }
            XCTAssertEqual(data.asArray(), [0, 0, 0, 0, 0])
            XCTAssertEqual(b.asArray(), [100, 0, 0, 0, 0])
        }
    }
    
#if !SWIFT_PACKAGE
    func testDeallocate() {
        print("Start")
        Thread.sleep(forTimeInterval: 5)
        do {
            var data = NDArrayData(value: 0, count: 1_000_000)
            data[0] = 100
            print("Allocate")
            Thread.sleep(forTimeInterval: 5)
        }
        print("Deallocate")
        Thread.sleep(forTimeInterval: 5)
    }
#endif
}
