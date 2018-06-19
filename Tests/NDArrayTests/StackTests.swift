import XCTest
import NDArray

class StackTests: XCTestCase {
    
    func testStack() {
        do {
            let a = NDArray.range(3)
            
            XCTAssertEqual(NDArray.stack([NDArray](repeating: a, count: 3)),
                           NDArray([[0, 1, 2],
                                    [0, 1, 2],
                                    [0, 1, 2]]))
            XCTAssertEqual(NDArray.stack([NDArray](repeating: a, count: 3), newAxis: -1),
                           NDArray([[0, 0, 0],
                                    [1, 1, 1],
                                    [2, 2, 2]]))
        }
    }

    func testConcat() {
        do {
            let a = NDArray.range(3)
            XCTAssertEqual(NDArray.concat([a, a], along: 0),
                           NDArray([0, 1, 2, 0, 1, 2]))
        }
        do {
            let a = NDArray([[0, 1],
                             [2, 3]])
            
            XCTAssertEqual(NDArray.concat([a, a], along: 0),
                           NDArray([[0, 1], [2, 3], [0, 1], [2, 3]]))
            XCTAssertEqual(NDArray.concat([a, a], along: -1),
                           NDArray([[0, 1, 0, 1], [2, 3, 2, 3]]))
        }
        do {
            let a = NDArray([[[0, 1],
                              [2, 3]],
                             [[4, 5],
                              [6, 7]]])
            
            XCTAssertEqual(NDArray.concat([a, a, a], along: 0),
                           NDArray([[[0, 1], [2, 3]],
                                    [[4, 5], [6, 7]],
                                    [[0, 1], [2, 3]],
                                    [[4, 5], [6, 7]],
                                    [[0, 1], [2, 3]],
                                    [[4, 5], [6, 7]]]))
            XCTAssertEqual(NDArray.concat([a, a, a], along: 1),
                           NDArray([[[0, 1], [2, 3], [0, 1], [2, 3], [0, 1], [2, 3]],
                                    [[4, 5], [6, 7], [4, 5], [6, 7], [4, 5], [6, 7]]]))
            XCTAssertEqual(NDArray.concat([a, a, a], along: -1),
                           NDArray([[[0, 1, 0, 1, 0, 1],
                                     [2, 3, 2, 3, 2, 3]],
                                    [[4, 5, 4, 5, 4, 5],
                                     [6, 7, 6, 7, 6, 7]]]))
        }
    }

}
