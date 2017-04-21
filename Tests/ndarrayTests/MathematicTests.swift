
import XCTest
@testable import NDArray

class MathematicTests: XCTestCase {

    func testAdd() {
        do {
            // scalar + scalar
            let a: NDArray = 1
            let b: NDArray = 2
            let c = a + b
            XCTAssertEqual(c.asScalar(), 3)
        }
        do {
            let a: NDArray = 1
            let b = NDArray.range(0..<24).reshaped([2, 3, 4])
            let c = a + b
            XCTAssertEqual(c, NDArray(shape: [2,3,4],
                                      elements: [ 1,  2,  3,  4,
                                                  5,  6,  7,  8,
                                                  9, 10, 11, 12,
                                                 13, 14, 15, 16,
                                                 17, 18, 19, 20,
                                                 21, 22, 23, 24]))
        }
        do {
            let a = NDArray.range(0..<8).reshaped([2, 2, 2])
            let b = NDArray.range(0..<4).reshaped([2, 2])
            
            XCTAssertEqual(a+b, NDArray(shape: [2, 2, 2], elements: [0, 2, 4, 6, 4, 6, 8, 10]))
            XCTAssertEqual(a[1]+b, NDArray(shape :[2, 2], elements: [4, 6, 8, 10]))
            
            let c = a.transposed()
            XCTAssertEqual(c+b, NDArray(shape: [2, 2, 2], elements: [0,  5,  4,  9,  1,  6,  5, 10]))
            XCTAssertEqual(c[1]+b, NDArray(shape: [2, 2], elements: [1, 6, 5, 10]))
            
            let d = b.transposed()
            XCTAssertEqual(c+d, NDArray(shape: [2, 2, 2], elements: [0,  6,  3,  9,  1,  7,  4, 10]))
            XCTAssertEqual(c[1]+d, NDArray(shape: [2, 2], elements: [1, 7, 4, 10]))
        }
        do {
            let a = NDArray(shape: [0, 2, 2], elements: [])
            let b = NDArray(shape: [2], elements: [1, 2])
            
            XCTAssertEqual(a+b, NDArray(shape: [0, 2, 2], elements: []))
        }
    }
    
    func testNeg() {
        do {
            // continuous
            let a = NDArray.range(8).reshaped([2, 2, 2])[1]
            let b = -a
            XCTAssertEqual(b, NDArray(shape: [2, 2], elements: [-4, -5, -6, -7]))
        }
        do {
            // dense, uncontinuous
            let a = NDArray.range(8).reshaped([2, 2, 2]).transposed()
            let b = -a
            XCTAssertEqual(b, NDArray(shape: [2, 2, 2], elements: [-0, -4, -2, -6, -1, -5, -3, -7]))
        }
        do {
            // not dense
            let a = NDArray.range(8).reshaped([2, 2, 2])[nil, 1]
            let b = -a
            XCTAssertEqual(b, NDArray(shape: [2, 2], elements: [-2, -3, -6, -7]))
        }
    }
    
    func testSqrt() {
        do {
            // continuous
            let a = NDArray.range(8).reshaped([2, 2, 2])[1]
            let b = sqrt(a)
            XCTAssertEqual(b, NDArray(shape: [2, 2], elements: [4, 5, 6, 7].map(sqrtf)))
        }
        do {
            // dense, uncontinuous
            let a = NDArray.range(8).reshaped([2, 2, 2]).transposed()
            let b = sqrt(a)
            XCTAssertEqual(b, NDArray(shape: [2, 2, 2], elements: [0, 4, 2, 6, 1, 5, 3, 7].map(sqrtf)))
        }
        do {
            // not dense
            let a = NDArray.range(8).reshaped([2, 2, 2])[nil, 1]
            let b = sqrt(a)
            XCTAssertEqual(b, NDArray(shape: [2, 2], elements: [2, 3, 6, 7].map(sqrtf)))
        }
    }
}
