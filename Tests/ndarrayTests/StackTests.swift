
import XCTest
@testable import NDArray

class StackTests: XCTestCase {

    func testConcat() {
        do {
            let a = NDArray.range(3)
            XCTAssertEqual(NDArray.concat([a, a], along: 0),
                           NDArray(shape: [6], elements: [0, 1, 2, 0, 1, 2]))
        }
        do {
            let a = NDArray(shape: [2, 2], elements: [0, 1, 2, 3])
            
            XCTAssertEqual(NDArray.concat([a, a], along: 0),
                           NDArray(shape: [4, 2], elements: [0, 1, 2, 3, 0, 1, 2, 3]))
            XCTAssertEqual(NDArray.concat([a, a], along: -1),
                           NDArray(shape: [2, 4], elements: [0, 1, 0, 1, 2, 3, 2, 3]))
        }
        do {
            let a = NDArray(shape: [2, 2, 2], elements: [0, 1, 2, 3, 4, 5, 6, 7])
            
            XCTAssertEqual(NDArray.concat([a, a, a], along: 0),
                           NDArray(shape: [6, 2, 2], elements: [0, 1, 2, 3, 4, 5, 6, 7,
                                                                0, 1, 2, 3, 4, 5, 6, 7,
                                                                0, 1, 2, 3, 4, 5, 6, 7]))
            XCTAssertEqual(NDArray.concat([a, a, a], along: 1),
                           NDArray(shape: [2, 6, 2], elements: [0, 1, 2, 3, 0, 1, 2, 3,
                                                                0, 1, 2, 3, 4, 5, 6, 7,
                                                                4, 5, 6, 7, 4, 5, 6, 7]))
            XCTAssertEqual(NDArray.concat([a, a, a], along: -1),
                           NDArray(shape: [2, 2, 6], elements: [0, 1, 0, 1, 0, 1,
                                                                2, 3, 2, 3, 2, 3,
                                                                4, 5, 4, 5, 4, 5,
                                                                6, 7, 6, 7, 6, 7]))
        }
    }

}
