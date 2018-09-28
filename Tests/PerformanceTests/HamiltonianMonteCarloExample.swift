import XCTest
import NDArray
import TestHelper

#if !DEBUG
class HamiltonianMonteCarloExample: XCTestCase {
    func testHamiltonianMonteCarlo() {
        
        let sampleNum = 10000
        
        let epsilon: Float = 0.1
        let steps = 30
        
        var state = NDArray([[0], [0]])
        
        var samples: [NDArray] = []
        
        // burn in
        for _ in 0..<sampleNum/10 {
            state = sample(state: state, epsilon: epsilon, steps: steps)
        }
        
        for _ in 0..<sampleNum {
            state = sample(state: state, epsilon: epsilon, steps: steps)
            samples.append(state)
        }
        
        let cat = NDArray.concat(samples, along: 1)
        
        XCTAssertEqual(mean(cat, along: 1), mu.squeezed(), accuracy: 0.1)
        XCTAssertEqual(cov(cat), sigma, accuracy: 0.1)
        
        print("Accepted: \(samples.count)/\(sampleNum)")
        
        // csv output
//        for row in cat.transposed(){
//            print("\(row[0].asScalar()), \(row[1].asScalar())")
//        }
    }
}
#endif

private let mu = NDArray([[30], [0]])
private let sigma = NDArray([[1, 0.5],
                            [0.5, 1]])
private var sigma_inv = try! inv(sigma)


private func log_normal(x: NDArray) -> NDArray {
    return -0.5 * (x-mu).transposed() |*| sigma_inv |*| (x-mu)
}

private func d_log_normal(x: NDArray) -> NDArray {
    return -sigma_inv |*| (x-mu)
}

private func hamiltonian(state: NDArray, velocity: NDArray) -> NDArray {
    return (velocity.transposed() |*| velocity)/2 - log_normal(x: state)
}

private func sample(state: NDArray, epsilon: Float, steps: Int) -> NDArray {
    let velocity = NDArray.normal(mu: 0, sigma: 1, shape: [2, 1])
    
    var new_state = state
    var new_velocity = velocity
    
    for _ in 0..<steps {
        // leap frog
        new_velocity += -epsilon/2 * -d_log_normal(x: new_state)
        new_state += epsilon * new_velocity
        new_velocity += -epsilon/2 * -d_log_normal(x: new_state)
    }
    
    let r = exp(hamiltonian(state: new_state, velocity: new_velocity) - hamiltonian(state: state, velocity: velocity))
    if r.asScalar() > NDArray.uniform(low: 0, high: 1, shape: []).asScalar() {
        return new_state
    } else {
        return state
    }
}
