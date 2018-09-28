import XCTest
@testable import NDArray

public func XCTAssertEqual(_ expression1: NDArray,
                    _ expression2: NDArray,
                    accuracy: Float,
                    file: StaticString = #file,
                    line: UInt = #line) {
    XCTAssertEqual(expression1.shape, expression2.shape, file: file, line: line)
    
    let elements1 = gatherElements(expression1)
    let elements2 = gatherElements(expression2)
    
    for (e1, e2) in zip(elements1, elements2) {
        XCTAssertEqual(e1, e2, accuracy: accuracy, file: file, line: line)
    }
}
