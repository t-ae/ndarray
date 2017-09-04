
import XCTest
import NDArray

class SubscriptTests: XCTestCase {

    func testSubscriptGet1() {
        do {
            let a = NDArray.range(0..<27).reshaped([3, 3, 3])
            XCTAssertEqual(a[], a)
            
            XCTAssertEqual(a[1], NDArray([[ 9, 10, 11],
                                          [12, 13, 14],
                                          [15, 16, 17]]))
            
            XCTAssertEqual(a[-2], a[1])
            
            XCTAssertEqual(a[1, 1], NDArray([12, 13, 14]))
            
            XCTAssertEqual(a[1, 1, 1].asScalar(), 13)
            
            XCTAssertEqual(a[nil, 1], NDArray([[ 3,  4,  5],
                                               [12, 13 ,14],
                                               [21, 22, 23]]))
            
            XCTAssertEqual(a[][nil, 1], NDArray([[ 3,  4,  5],
                                                 [12, 13 ,14],
                                                 [21, 22, 23]]))

            XCTAssertEqual(a[1..<2, 1..<3], NDArray([[[12, 13, 14],
                                                      [15, 16, 17]]]))
            
            XCTAssertEqual(a[1..<2][nil, 1..<3], NDArray([[[12, 13, 14],
                                                           [15, 16, 17]]]))
            
            XCTAssertEqual(a[0..<0], NDArray.zeros([0, 3, 3]))
        }
        do {
            // uncontiguous
            let a = NDArray.range(0..<27).reshaped([3, 3, 3]).transposed()
            
            XCTAssertEqual(a[1], NDArray([[ 1, 10, 19],
                                          [ 4, 13, 22],
                                          [ 7, 16, 25]]))
            
            XCTAssertEqual(a[1, 1], NDArray([4, 13, 22]))
            
            XCTAssertEqual(a[1, 1, 1].asScalar(), 13)
            
            XCTAssertEqual(a[nil, 1], NDArray([[ 3, 12, 21],
                                               [ 4, 13, 22],
                                               [ 5, 14, 23]]))
        }
        do {
            // uncontiguous + offset
            let a = NDArray.range(0..<27).reshaped([3, 3, 3]).transposed()[1]
            
            XCTAssertEqual(a[1], NDArray([4, 13, 22]))
            
            XCTAssertEqual(a[1, 1].asScalar(), 13)
            
            XCTAssertEqual(a[nil, 1], NDArray([10, 13, 16]))

        }
    }
    
    func testSubscriptSet1() {
        do {
            // contiguous and contiguous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let b = NDArray.range(0..<12).reshaped([3, 4]) + 50
            a[1] = b
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[50, 51, 52, 53],
                                        [54, 55, 56, 57],
                                        [58, 59, 60, 61]]]))
            
            a[1][1..<3, 1..<3] = NDArray([-1, -2])
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[50, 51, 52, 53],
                                        [54, -1, -2, 57],
                                        [58, -1, -2, 61]]]))
        }
        do {
            // uncontiguous and contiguous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let b = NDArray.range(0..<8).reshaped([2, 4]) + 50
            a[nil, 1] = b
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [50, 51, 52, 53],
                                        [ 8,  9, 10, 11]],
                                       [[12, 13, 14, 15],
                                        [54, 55, 56, 57],
                                        [20, 21, 22, 23]]]))
        }
        do {
            // contiguous and uncontiguous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let b = (NDArray.range(0..<24).reshaped([2, 3, 4]) + 50)[1, nil, 1]
            // b = shape: [3], elements: [63, 67, 71]
            a[nil, nil, 1] = b
            XCTAssertEqual(a, NDArray([[[ 0, 63,  2,  3],
                                        [ 4, 67,  6,  7],
                                        [ 8, 71, 10, 11]],
                                       [[12, 63, 14, 15],
                                        [16, 67, 18, 19],
                                        [20, 71, 22, 23]]]))
            
        }
        do {
            // contiguous and uncontiguous
            var a = NDArray.range(0..<24).reshaped([2, 3, 4])
            let zero: NDArray = NDArray(scalar: 0)
            a[1, 1, 2] = zero
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[12, 13, 14, 15],
                                        [16, 17,  0, 19],
                                        [20, 21, 22, 23]]]))
        }
        do {
            // uncontiguous and uncontiguous
            var a = NDArray.range(0..<8).reshaped([2, 2, 2]).transposed()
            let b = NDArray.range(0..<4).reshaped([2, 2]).transposed() + 1
            a[1] = b
            XCTAssertEqual(a, NDArray([[[0, 4],
                                        [2, 6]],
                                       [[1, 3],
                                        [2, 4]]]))
        }
        do {
            var a = NDArray.range(0..<8).reshaped([2, 2, 2]).transposed()[1]
            // a = shape:[2, 2], elements = [1, 5, 3, 7]
            let b = NDArray.range(0..<4).reshaped([2, 2]).transposed()[1]
            // b = shape: [2], elements = [1, 3]
            a[] = b
            XCTAssertEqual(a, NDArray([[1, 3],
                                       [1, 3]]))
        }
        do {
            var a = NDArray.range(4).reshaped([2, 2])
            a[] = NDArray([5, 6]).flipped(0)
            XCTAssertEqual(a, NDArray([[6, 5],
                                       [6, 5]]))
            a[] = NDArray([[5], [6]]).flipped(0)
            XCTAssertEqual(a, NDArray([[6, 6],
                                       [5, 5]]))
        }
        
        do {
            // CoW
            let a = NDArray.range(4)
            var b = a
            b[0] = NDArray(scalar: 100)
            XCTAssertEqual(a, NDArray([0, 1, 2, 3]))
            XCTAssertEqual(b, NDArray([100, 1, 2, 3]))
        }
    }
    
    func testSubscriptGet2() {
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4])
            let b = a[]
            XCTAssertEqual(b, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[12, 13, 14, 15],
                                        [16, 17, 18, 19],
                                        [20, 21, 22, 23]]]))
            
            let c = b[nil, 1..<3]
            XCTAssertEqual(c, NDArray([[[ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[16, 17, 18, 19],
                                        [20, 21, 22, 23]]]))
            
            let d = c[1, nil, ..<3~>-2]
            XCTAssertEqual(d, NDArray([[18, 16],
                                       [22, 20]]))
            
            let e = c[1, nil, 1..<4~>-2]
            XCTAssertEqual(e, NDArray([[19, 17],
                                       [23, 21]]))
            
            let f = c[1, nil, 0...~>-3]
            XCTAssertEqual(f, NDArray([[19, 16],
                                       [23, 20]]))
        }
        do {
            let a = NDArray.range(24).reshaped([2, 3, 4]).flipped(0).flipped(1).flipped(2)
            let b = a[nil]
            XCTAssertEqual(b, NDArray([[[23, 22, 21, 20],
                                        [19, 18, 17, 16],
                                        [15, 14, 13, 12]],
                                       [[11, 10,  9,  8],
                                        [ 7,  6,  5,  4],
                                        [ 3,  2,  1,  0]]]))
            
            let c = b[nil, ~>2]
            XCTAssertEqual(c, NDArray([[[23, 22, 21, 20],
                                        [15, 14, 13, 12]],
                                       [[11, 10,  9,  8],
                                        [ 3,  2,  1,  0]]]))
            
            let d = c[nil, nil, 1..<4~>-1]
            XCTAssertEqual(d, NDArray([[[20, 21, 22],
                                        [12, 13, 14]],
                                       [[ 8,  9, 10],
                                        [ 0,  1,  2]]]))
            
            let e = d[1]
            XCTAssertEqual(e, NDArray([[ 8,  9, 10],
                                       [ 0,  1,  2]]))
        }
    }
    
    func testSubscriptSet2() {
        do {
            var a = NDArray([0, 1, 2, 3, 4, 5, 6])
            a[~>-2] = NDArray([-6, -4, -2, 0])
            XCTAssertEqual(a, NDArray([0, 1, -2, 3, -4, 5, -6]))

            a[..<6~>-2] = NDArray([-6, -4, -2])
            XCTAssertEqual(a, NDArray([0, -2, -2, -4, -4, -6, -6]))
        }
        do {
            var a = NDArray.range(24).reshaped([2, 3, 4])
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[12, 13, 14, 15],
                                        [16, 17, 18, 19],
                                        [20, 21, 22, 23]]]))
            
            a[1] = NDArray(scalar: 0)
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[ 0,  0,  0,  0],
                                        [ 0,  0,  0,  0],
                                        [ 0,  0,  0,  0]]]))
            
            a[1, ~>2, 1...~>2] = NDArray([1, 2])
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[ 0,  1,  0,  2],
                                        [ 0,  0,  0,  0],
                                        [ 0,  1,  0,  2]]]))
            
            a[1, 1, 0..<3~>-2] = NDArray([1, 2])
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[ 0,  1,  0,  2],
                                        [ 2,  0,  1,  0],
                                        [ 0,  1,  0,  2]]]))
            
            a[1, 1, ~>-2] = NDArray([-1, -2]).flipped(0)
            XCTAssertEqual(a, NDArray([[[ 0,  1,  2,  3],
                                        [ 4,  5,  6,  7],
                                        [ 8,  9, 10, 11]],
                                       [[ 0,  1,  0,  2],
                                        [ 2, -1,  1, -2],
                                        [ 0,  1,  0,  2]]]))
        }
        do {
            var a = NDArray.range(24).reshaped([2, 3, 4]).flipped(0).flipped(1).flipped(2)
            XCTAssertEqual(a, NDArray([[[23, 22, 21, 20],
                                        [19, 18, 17, 16],
                                        [15, 14, 13, 12]],
                                       [[11, 10,  9,  8],
                                        [ 7,  6,  5,  4],
                                        [ 3,  2,  1,  0]]]))
            
            a[1] = NDArray([1, 2, 3, 4])
            XCTAssertEqual(a, NDArray([[[23, 22, 21, 20],
                                        [19, 18, 17, 16],
                                        [15, 14, 13, 12]],
                                       [[ 1,  2,  3,  4],
                                        [ 1,  2,  3,  4],
                                        [ 1,  2,  3,  4]]]))
        }
    }
}
