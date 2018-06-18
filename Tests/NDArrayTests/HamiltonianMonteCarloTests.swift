
import XCTest
import NDArray

class HamiltonianMonteCarloTests: XCTestCase {

    static let mu = NDArray([[5], [0]])
    static let sigma = NDArray([[1, 0.5],
                         [0.5, 1]])
    static var sigma_inv = try! inv(sigma)
    
    var mu: NDArray {
        return HamiltonianMonteCarloTests.mu
    }
    var sigma: NDArray {
        return HamiltonianMonteCarloTests.sigma
    }
    var sigma_inv: NDArray {
        return HamiltonianMonteCarloTests.sigma_inv
    }
    
    func log_normal(x: NDArray) -> NDArray {
        return -0.5 * (x-mu).transposed() |*| sigma_inv |*| (x-mu)
    }
    
    func d_log_normal(x: NDArray) -> NDArray {
        return -sigma_inv |*| (x-mu)
    }
    func hamiltonian(state: NDArray, velocity: NDArray) -> NDArray {
        return (velocity.transposed() |*| velocity)/2 - log_normal(x: state)
    }
    
    func sample(state: NDArray, epsilon: Float, tau: Float, steps: Int) -> NDArray {
        let velocity = NDArray.normal(mu: 0, sigma: tau, shape: [2, 1])
        
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
    
    func testHamiltonianMonteCarlo() {
        let epsilon: Float = 0.1
        let tau: Float = 2.0
        let steps = 30
        
        var state = NDArray([[0], [0]])
        
        var samples: [NDArray] = []
        
        for _ in 0..<1000 {
            state = sample(state: state, epsilon: epsilon, tau: tau, steps: steps)
            samples.append(state)
        }
        
        let cat = NDArray.concat(samples, along: 1)
        for row in cat.transposed(){
            //print("\(row[0].asScalar()), \(row[1].asScalar())")
        }
    }

}
