
import XCTest
import NDArray

class IrisClassificationTests: XCTestCase {

    func testLogisticRegression() {
        
        // TODO: Not completed yet
        
        let x = Iris.x_train
        let y = toOneHot(Iris.y_train)
        
        let numFeatures = x.shape[1]
        let numTrain = x.shape[0]
        
        let numOutput = y.shape[1]
        
        // Two layer neural network
        
        let numHiddenUnits1 = 5
        
        var W1 = NDArray.normal(mu: 0, sigma: 0.2, shape: [numFeatures, numHiddenUnits1])
        var b1 = NDArray.zeros([numHiddenUnits1])
        
        var W2 = NDArray.normal(mu: 0, sigma: 0.2, shape: [numHiddenUnits1, numOutput])
        var b2 = NDArray.zeros([numOutput])
        
        let alpha: Float = 1e-3
        
        for i in 0..<1 {
            let h1_1 = x <*> W1     // [M, 5]
            let h1_2 = h1_1 + b1    // [M, 5]
            let h1 = relu(h1_2)     // [M, 5]
            
            let h2_1 = h1 <*> W2    // [M, 3]
            let h2 = h2_1 + b2      // [M, 3]
            
            let out = softmax(h2)   // [M, 3]
            
            // back propagation
            let d_out_h2 = NDArray.diagonal(out - y)  // [90, 3, 3]
            
            let d_h2_b2 = NDArray.ones(b2.shape + [numOutput])        // [90, 3, 3]
            let d_h2_h2_1 = NDArray.ones(h2_1.shape + [numOutput])    // [90, 3, 3]
            
            let d_h2_1_W2 = NDArray.stack([NDArray](repeating: h1, count: numFeatures), newAxis: -1) // [90, 5, 3]
            let d_h2_1_h1 = W2
            
            let d_h1_h1_2 = relu(h1_2)
            
            let d_h1_2_b1 = NDArray.ones(b1.shape)
            let d_h1_2_h1_1 = NDArray.ones(h1_1.shape)
            
            let d_h1_1_W1 = NDArray.stack([NDArray](repeating: x, count: numHiddenUnits1), newAxis: -1)
            
            print("d_out_h2: \(d_out_h2.shape)")
            print("d_h2_b2: \(d_h2_b2.shape)")
            print("d_h2_h2_1: \(d_h2_h2_1.shape)")
            print("d_h2_1_W2: \(d_h2_1_W2.shape)")
            print("d_h2_1_h1: \(d_h2_1_h1.shape)")
            print("d_h1_h1_2: \(d_h1_h1_2.shape)")
            print("d_h1_2_b1: \(d_h1_2_b1.shape)")
            print("d_h1_2_h1_1: \(d_h1_2_h1_1.shape)")
            print("d_h1_1_W1: \(d_h1_1_W1.shape)")
            
            // chain
            let d_out_h2_1 = d_out_h2 * d_h2_h2_1                      // [90, 3]
            let d_out_h1 = d_out_h2_1.reshaped([90, 1, 3]) * d_h2_1_h1 // [90, 5, 3]
            let d_out_h1_2 = d_out_h1 * d_h1_h1_2 // [90, 5, 3]
            let d_out_h1_1 = d_out_h1_2 * d_h1_2_h1_1
            
            // update
            b2 -= alpha * mean(d_out_h2 * d_h2_b2, along: 0)
            W2 -= alpha * mean(d_out_h2_1.reshaped([90, 1, 3]) * d_h2_1_W2, along: 0)
            
            b1 -= alpha * mean(d_out_h1_2 * d_h1_2_b1, along: 0)
            W1 -= alpha * mean(d_out_h1_1 * d_h1_1_W1, along: 0)
            
            if i%100 == 0 {
                print("step: \(i)")
                let losses = -y * log(max(y - out, 1e-10))
                let loss = mean(losses, along: 0)
                print("loss: \(loss)")
            }
            
        }
    }

}

func sigmoid(_ x: NDArray) -> NDArray {
    return 1 / (1 + exp(-x))
}

func relu(_ x: NDArray) -> NDArray {
    return max(x, 0)
}

func softmax(_ x: NDArray) -> NDArray {
    var ret: [NDArray] = []

    for i in 0..<x.shape[0] {
        let row = exp(x[i])
        ret.append(row / sum(row))
    }
    
    return NDArray.stack(ret)
}

func toOneHot(_ y: NDArray) -> NDArray {
    precondition(y.ndim == 1)
    
    let size = Int(max(y).asScalar()) + 1
    
    var ret: [NDArray] = []
    
    for i in y.elements() {
        var vector = NDArray.zeros([size])
        vector[Int(i)] = NDArray(scalar: 1)
        ret.append(vector)
    }
    
    return NDArray.stack(ret)
}
