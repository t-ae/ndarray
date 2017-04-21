
import XCTest
@testable import NDArray

class GradientDescentTests: XCTestCase {

    func testGradientDescent() {
        // y = 3*x^2 + 2*x + 1
        
        let xs = NDArray.linspace(low: -2, high: 2, count: 300).reshaped([-1, 1])
        let ys = 3*xs*xs + 2*xs + 1 + NDArray.uniform(low: -0.1, high: 0.1, shape: xs.shape)
        
        print("xs: \(xs.shape), ys: \(ys.shape)")
        
        let features = NDArray.concat([xs*xs, xs, NDArray.ones(xs.shape)], along: 1)
        print("features: \(features.shape)")
        
        var theta = NDArray(shape: [3], elements: [1, 1, 1])
        
        let alpha: Float = 0.001
        
        for i in 0..<100 {
            print("\nstep: \(i)")
            let distance = sum(theta * features, along: 1) - ys.raveled() // shape: [300]
            let loss = mean(exp2(distance), along: 0) / 2
            print("loss: \(loss.asScalar())")
            
            let update = alpha * mean(distance.reshaped([-1, 1]) * features, along: 0)
            print("update: \(update)")
            theta -= update
            print("theta: \(theta)")
        }
        
        print(theta)
        
    }
    
    
}
