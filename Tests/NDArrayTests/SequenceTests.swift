
import XCTest
import NDArray

class SequenceTests: XCTestCase {

    func testSequence() {
        do {
            let vectors = NDArray([[0, 0, 0],
                                   [1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1],
                                   [1, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 1],
                                   [1, 1, 1]])
            let es = NDArray.stack(vectors.filter { vectorNorm($0).asScalar() == 1 })
            XCTAssertEqual(es, NDArray([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]]))
        }
    }
}
