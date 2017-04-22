
import XCTest

func XCTAssertEqualWithAccuracy(_ expression1: NDArray, _ expression2: NDArray, accuracy: Float) {
    XCTAssertEqual(expression1.shape, expression2.shape)
    
    let elements1 = gatherElements(expression1)
    let elements2 = gatherElements(expression2)
    for (e1, e2) in zip(elements1, elements2) {
        if fabsf(e1-e2) > accuracy {
            XCTFail("Assertion failed:\n\(elements1)\n\(elements2)")
        }
    }
}
