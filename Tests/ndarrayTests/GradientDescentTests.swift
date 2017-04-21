
import XCTest
@testable import NDArray

class GradientDescentTests: XCTestCase {

    func testGradientDescent() {
        // y = 0.3*x^2 + 0.2*x + 0.1
        
        let xs = NDArray.linspace(low: -1, high: 1, count: 300).reshaped([-1, 1])
        
        // data
        var ys = 0.3*xs*xs + 0.2*xs + 0.1
        ys += NDArray.normal(mu: 0, sigma: 0.03, shape: xs.shape)
        ys = ys.raveled()
        
        print("xs: \(xs.shape), ys: \(ys.shape)")
        
        // x^2, x^1, x^0
        let features = NDArray.concat([xs*xs, xs, NDArray.ones(xs.shape)], along: 1)
        print("features: \(features.shape)")
        
        var theta = NDArray(shape: [3], elements: [1, 1, 1])
        
        let alpha: Float = 0.1
        
        for i in 0..<10000 {
            print("\nstep: \(i)")
            
            // calculate loss
            let distance = sum(theta * features, along: 1) - ys
            let loss = mean(distance*distance, along: 0) / 2
            print("loss: \(loss.asScalar())")
            
            // Update parameters
            let grads = distance.reshaped([-1, 1]) * features
            print("grads: \(grads.shape)")
            let update = alpha * mean(grads, along: 0)
            print("update: \(update)")
            theta -= update
            print("theta: \(theta)")
        }
        
        print("\nanswer")
        print(theta)
        let distance = sum(theta * features, along: 1) - ys
        let loss = mean(distance*distance, along: 0) / 2
        print("loss: \(loss.asScalar())")
        print("")
        
    }
    
    
}
