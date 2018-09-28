import Foundation
import Xorswift

extension NDArray {
    /// Create randomly initialized NDArray.
    ///
    /// All values are sampled from the interval [`low`, `high`).
    public static func uniform(low: Float = 0, high: Float = 1, shape: [Int]) -> NDArray {
        precondition(shape.all { $0 >= 0 }, "Invalid shape: \(shape)")
        precondition(low < high, "low(\(low) is not less than high(\(high)))")
        
        let size = shape.prod()
        var buf = [Float](repeating: 0, count: size)
        
        buf.withUnsafeMutableBufferPointer {
            var gen = XorshiftGenerator()
            gen.fillUniform($0, with: low..<high)
        }
        
        return NDArray(shape: shape, elements: buf)
    }
    
    /// Create randomly initialized NDArray.
    ///
    /// All elements are sampled from N(mu, sigma).
    public static func normal(mu: Float = 0, sigma: Float = 1, shape: [Int]) -> NDArray {
        precondition(shape.all { $0 >= 0 }, "Invalid shape: \(shape)")
        precondition(sigma >= 0, "sigma < 0")
        
        let size = shape.prod()
        var buf = [Float](repeating: 0, count: size)
        buf.withUnsafeMutableBufferPointer {
            var gen = XorshiftGenerator()
            gen.fillNormal($0, mu: mu, sigma: sigma)
        }
        
        return NDArray(shape: shape, elements: buf)
    }
}
