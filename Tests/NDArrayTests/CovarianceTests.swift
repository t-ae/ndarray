import XCTest
@testable import NDArray
import TestHelper

class CovarianceTests: XCTestCase {

    func testCovariance() {
        
        let matrix = cov(Iris.x_train.transposed())
        
        XCTAssertEqual(matrix, matrix.transposed())
        
        let answer = NDArray([[ 0.78304317, -0.02713582,  1.47912342,  0.59903701],
                              [-0.02713582,  0.18066173, -0.27110865, -0.10254815],
                              [ 1.47912342, -0.27110865,  3.45583208,  1.44405182],
                              [ 0.59903701, -0.10254815,  1.44405182,  0.6398222 ]])
        
        XCTAssertEqual(matrix, answer, accuracy: 1e-5)
    }
    
}
