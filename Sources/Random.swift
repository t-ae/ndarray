
import Foundation

private func _uniform(low: Float = 0, high: Float = 1) -> Float {
    return (high-low)*(Float(arc4random_uniform(UInt32.max)) / Float(UInt32.max))+low
}

extension NDArray {
    public static func uniform(low: Float = 0, high: Float = 1, shape: [Int]) -> NDArray {
        let count = shape.prod()
        let elements = (0..<count).map { _ in _uniform(low: low, high: high) }
        
        return NDArray(shape: shape, elements: elements)
    }
    
    public static func normal(mu: Float = 0, sigma: Float = 0, shape: [Int]) -> NDArray {
        let u1 = uniform(low: 0, high: 1, shape: shape)
        let u2 = uniform(low: 0, high: 1, shape: shape)
        
        let stdNormal =  sqrt(-2*log(u1)) * cos(2*Float.pi*u2)
        
        return stdNormal*sigma + mu
    }
}
