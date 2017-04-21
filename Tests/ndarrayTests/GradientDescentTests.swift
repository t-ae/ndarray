
import XCTest
@testable import NDArray

class GradientDescentTests: XCTestCase {

    func testGradientDescent() {
        // y = 3*x^2 + 2*x + 1
        
        let xs = NDArray.linspace(low: -2, high: 2, count: 300)
        let ys = 3*xs*xs + 2*xs + 1 + NDArray.uniform(low: -0.1, high: 0.1, shape: xs.shape)
        
        var features = NDArray.stack([xs*xs, xs, NDArray.ones(xs.shape)])
        
        
        
    }
    
    
}
