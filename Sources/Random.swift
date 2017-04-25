
import Foundation

private func _uniform(low: Float = 0, high: Float = 1) -> Float {
    return (high-low) * (Float(arc4random_uniform(UInt32.max-1)) / Float(UInt32.max)) + low
}

extension NDArray {
    /// Create randomly initialized NDArray.
    ///
    /// All values are sampled from the interval [`low`, `high`).
    public static func uniform(low: Float = 0, high: Float = 1, shape: [Int]) -> NDArray {
        let count = shape.prod()
        let elements = (0..<count).map { _ in _uniform(low: low, high: high) }
        
        return NDArray(shape: shape, elements: elements)
    }
    
    /// Create randomly initialized NDArray.
    ///
    /// All elements are sampled from N(mu, sigma).
    public static func normal(mu: Float = 0, sigma: Float = 0, shape: [Int]) -> NDArray {
        // Box-Muller's method
        let u1 = uniform(low: 0, high: 1, shape: shape)
        let u2 = uniform(low: 0, high: 1, shape: shape)
        
        let stdNormal =  sqrt(-2*log(u1)) * cos(2*Float.pi*u2)
        
        return stdNormal*sigma + mu
    }
}
