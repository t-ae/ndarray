
import XCTest
@testable import NDArray

class SubscriptTests: XCTestCase {

    func testSubscriptGet() {
        do {
            let a = NDArray.range(0..<27).reshaped([3, 3, 3])
            
            XCTAssertEqual(a[1], NDArray(shape: [3, 3], elements: [9, 10, 11, 12, 13, 14, 15, 16, 17]))
            
            XCTAssertEqual(a[1, 1], NDArray(shape: [3], elements: [12, 13, 14]))
            
            XCTAssertEqual(a[1, 1, 1].asScalar(), 13)
            
            XCTAssertEqual(a[nil, 1], NDArray(shape: [3, 3], elements: [3, 4, 5, 12, 13 ,14, 21, 22, 23]))
        }
        do {
            // uncontinuous
            let a = NDArray.range(0..<27).reshaped([3, 3, 3]).transposed()
            
            XCTAssertEqual(a[1], NDArray(shape: [3, 3], elements: [1, 10, 19, 4, 13, 22, 7, 16, 25]))
            
            XCTAssertEqual(a[1, 1], NDArray(shape: [3], elements: [4, 13, 22]))
            
            XCTAssertEqual(a[1, 1, 1].asScalar(), 13)
            
            XCTAssertEqual(a[nil, 1], NDArray(shape: [3, 3], elements:[3, 12, 21, 4, 13, 22, 5, 14, 23]))
        }
        do {
            // uncontinuous + offset
            let a = NDArray.range(0..<27).reshaped([3, 3, 3]).transposed()[1]
            
            XCTAssertEqual(a[1], NDArray(shape: [3], elements: [4, 13, 22]))
            
            XCTAssertEqual(a[1, 1].asScalar(), 13)
            
            XCTAssertEqual(a[nil, 1], NDArray(shape: [3], elements: [10, 13, 16]))

        }
    }
    
    func testSubscriptSet() {
        do {
            // continuous and continuous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let b = NDArray.range(0..<12).reshaped([3, 4]) + 50
            a[1] = b
            XCTAssertEqual(a, NDArray(shape: [2, 3, 4],
                                      elements: [ 0,  1,  2,  3,
                                                  4,  5,  6,  7,
                                                  8,  9, 10, 11,
                                                 50, 51, 52, 53,
                                                 54, 55, 56, 57,
                                                 58, 59, 60, 61]))
        }
        do {
            // uncontinuous and continuous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let b = NDArray.range(0..<8).reshaped([2, 4]) + 50
            a[nil, 1] = b
            XCTAssertEqual(a, NDArray(shape: [2, 3, 4],
                                      elements: [ 0,  1,  2,  3,
                                                 50, 51, 52, 53,
                                                  8,  9, 10, 11,
                                                 12, 13, 14, 15,
                                                 54, 55, 56, 57,
                                                 20, 21, 22, 23]))
        }
        do {
            // continuous and uncontinuous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let b = (NDArray.range(0..<24).reshaped([2, 3, 4]) + 50)[1, nil, 1]
            // b = shape: [3], elements: [63, 67, 71]
            a[nil, nil, 1] = b
            XCTAssertEqual(a, NDArray(shape: [2, 3, 4],
                                      elements: [ 0, 63,  2,  3,
                                                  4, 67,  6,  7,
                                                  8, 71, 10, 11,
                                                 12, 63, 14, 15,
                                                 16, 67, 18, 19,
                                                 20, 71, 22, 23]))
            
        }
        do {
            // continuous and uncontinuous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let zero: NDArray = NDArray(scalar: 0)
            a[1, 1, 2] = zero
            XCTAssertEqual(a, NDArray(shape: [2, 3, 4],
                                      elements: [ 0,  1,  2,  3,
                                                  4,  5,  6,  7,
                                                  8,  9, 10, 11,
                                                 12, 13, 14, 15,
                                                 16, 17,  0, 19,
                                                 20, 21, 22, 23]))
        }
        do {
            // uncontinuous and uncontinuous
            var a = NDArray.range(0..<8).reshaped([2, 2, 2]).transposed()
            let b = NDArray.range(0..<4).reshaped([2, 2]).transposed() + 1
            a[1] = b
            XCTAssertEqual(a, NDArray(shape: [2, 2, 2], elements: [0, 4, 2, 6, 1, 3, 2, 4]))
        }
        do {
            var a = NDArray.range(0..<8).reshaped([2, 2, 2]).transposed()[1]
            // a = shape:[2, 2], elements = [1, 5, 3, 7]
            let b = NDArray.range(0..<4).reshaped([2, 2]).transposed()[1]
            // b = shape: [2], elements = [1, 3]
            a[nil] = b
            XCTAssertEqual(a, NDArray(shape: [2, 2], elements: [1, 3, 1, 3]))
            
        }
    }
    
    func testHoge() {
        var a = NDArray.range(0..<8).reshaped([2,2,2])
        let zero: NDArray = NDArray(scalar: 0)
        a[1] = zero
    }
}
