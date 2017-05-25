
import XCTest
import NDArray

#if !SWIFT_PACKAGE
class IrisClassificationTests: XCTestCase {

    func testLogisticRegression() {
        
        // Equivalent python script on ./Util/iris_test.py
        
        let start = Date()
        
        let (normalize_mu, variance) = moments(Iris.x_train, along: 0)
        let normalize_sigma = sqrt(variance)
        
        let x = (Iris.x_train - normalize_mu) / normalize_sigma
        let labels = Iris.y_train
        let y = toOneHot(labels)
        
        let numFeatures = x.shape[1]
        let numTrainSamples = x.shape[0]
        let numOutput = y.shape[1]
        
        let labelsCount = sum(y, along: 0)
        
        // Two layer neural network
        // Input(4) -> Dense(5) -> ReLU -> Dense(3) -> Softmax
        
        let numHiddenUnits1 = 5
        
        // init with glorot uniform
        let W1_limit = sqrtf(6 / Float(numFeatures + numHiddenUnits1))
        var W1 = NDArray.uniform(low: -W1_limit, high: W1_limit, shape: [numFeatures, numHiddenUnits1]) // [4, 5]
        var b1 = NDArray.zeros([numHiddenUnits1]) // [5]
        
        let W2_limit = sqrtf(2 / Float(numHiddenUnits1 + numOutput))
        var W2 = NDArray.uniform(low: -W2_limit, high: W2_limit, shape: [numHiddenUnits1, numOutput]) // [5, 3]
        var b2 = NDArray.zeros([numOutput]) // [5]
        
        let alpha: Float = 1e-3
        
        for i in 0...30000 {
            let h1_1 = x |*| W1     // [90, 5]
            let h1_2 = h1_1 + b1    // [90, 5]
            let h1 = relu(h1_2)     // [90, 5]
            
            let h2_1 = h1 |*| W2    // [90, 3]
            let h2 = h2_1 + b2      // [90, 3]
            
            let out = softmax(h2)   // [90, 3]
            
            // back propagation
            let d_out_h2 = out - y  // [90, 3]
            
            let d_h2_b2 = NDArray.ones(b2.shape)        // [90, 3]
            let d_h2_h2_1 = NDArray.ones(h2_1.shape)    // [90, 3]
            
            let d_h2_1_W2 = h1 // [90, 5]
            let d_h2_1_h1 = W2 // [90, 5, 3]
            
            let d_h1_h1_2 = d_relu(h1_2) // [90, 5]
            
            let d_h1_2_b1 = NDArray.ones(b1.shape)      // [90, 5]
            let d_h1_2_h1_1 = NDArray.ones(h1_1.shape)  // [90, 5]
            
            let d_h1_1_W1 = x // [90, 4]
            
            // chain
            let d_out_b2 = d_h2_b2 * d_out_h2           // [90, 3]
            let d_out_h2_1 = d_h2_h2_1 * d_out_h2       // [90, 3]
            let d_out_W2 = d_h2_1_W2.expandDims(-1)
                |*| d_out_h2_1.expandDims(1) // [90, 5, 3]
            let d_out_h1 = (d_h2_1_h1 |*| d_out_h2_1.expandDims(-1))
                .squeeze()                              // [90, 5]
            let d_out_h1_2 = d_h1_h1_2 * d_out_h1       // [90, 5]
            let d_out_b1 = d_out_h1_2 * d_h1_2_b1       // [90, 5]
            let d_out_h1_1 = d_h1_2_h1_1 * d_out_h1_2   // [90, 5]
            let d_out_W1 = d_h1_1_W1.expandDims(-1)
                |*| d_out_h1_1.expandDims(1) // [90, 4, 5]
            
            // update
            b2 -= alpha * mean(d_out_b2, along: 0)
            W2 -= alpha * mean(d_out_W2, along: 0)
            
            b1 -= alpha * mean(d_out_b1, along: 0)
            W1 -= alpha * mean(d_out_W1, along: 0)
            
            if i%100 == 0 {
                print("\nstep: \(i)")
                let losses = -y * log(out.clipped(low: 1e-10))
                let loss = mean(sum(losses, along: 1)).asScalar()
                let featureLosses = sum(losses, along: 0) / labelsCount
                print("loss: \(loss), (\(featureLosses.elements()))")
                
                let answer = argmax(out, along: 1)
                let trues = zip(answer, labels).filter { return $0 == $1 }.count
                let accuracy = Float(trues) / Float(numTrainSamples)
                
                print("accuracy: \(accuracy)")
            }
        }
        
        // test
        do {
            let x = (Iris.x_test - normalize_mu) / normalize_sigma
            let labels = Iris.y_test
            let y = toOneHot(labels)
            let labelsCount = sum(y, along: 0)
            
            let h1_1 = x |*| W1     // [90, 5]
            let h1_2 = h1_1 + b1    // [90, 5]
            let h1 = relu(h1_2)     // [90, 5]
            
            let h2_1 = h1 |*| W2    // [90, 3]
            let h2 = h2_1 + b2      // [90, 3]
            
            let out = softmax(h2)   // [90, 3]
            
            print("\ntest result:")
            let losses = -y * log(out.clipped(low: 1e-10))
            let loss = mean(sum(losses, along: 1)).asScalar()
            let featureLosses = sum(losses, along: 0) / labelsCount
            print("loss: \(loss), (\(featureLosses.elements()))")
            let answer = argmax(out, along: 1)
            let trues = zip(answer, labels).filter { return $0 == $1 }.count
            let accuracy = Float(trues) / Float(x.shape[0])
            
            print("accuracy: \(accuracy)")
            print("")
        }
        
        print("elapsed time: \(Date().timeIntervalSince(start))sec\n")
    }
}

func relu(_ x: NDArray) -> NDArray {
    return x.clipped(low: 0)
}

func d_relu(_ x: NDArray) -> NDArray {
    return copySign(magnitude: 1, sign: x).clipped(low: 0)
}

func softmax(_ x: NDArray) -> NDArray {
    
    let e = exp(x)
    let eSum = sum(e, along: 1).reshaped([-1 ,1])

    return e / eSum
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

#endif
