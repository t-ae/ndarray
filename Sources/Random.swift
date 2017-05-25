
import Foundation
import Accelerate

extension NDArray {
    /// Create randomly initialized NDArray.
    ///
    /// All values are sampled from the interval [`low`, `high`).
    public static func uniform(low: Float = 0, high: Float = 1, shape: [Int]) -> NDArray {
        precondition(shape.all { $0 >= 0 })
        let size = shape.prod()
        var dst1 = [UInt32](repeating: 0, count: size)
        var dst2 = NDArrayData(size: size)
        arc4random_buf(&dst1, MemoryLayout<UInt32>.size * size)
        
        
        dst2.withUnsafeMutablePointer {
            vDSP_vfltu32(dst1, 1, $0, 1, vDSP_Length(size))
            
            var factor = (high - low) / Float(UInt32.max)
            var low = low
            vDSP_vsmsa($0, 1, &factor, &low, $0, 1, vDSP_Length(size))
        }
        
        return NDArray(shape: shape, elements: dst2)
        
    }
    
    /// Create randomly initialized NDArray.
    ///
    /// All elements are sampled from N(mu, sigma).
    public static func normal(mu: Float = 0, sigma: Float = 1, shape: [Int]) -> NDArray {
        // Box-Muller's method
        let u1 = uniform(low: 0, high: 1, shape: shape)
        let u2 = uniform(low: 0, high: 1, shape: shape)
        
        let stdNormal =  sqrt(-2*log(u1)) * cos(2*Float.pi*u2)
        
        return stdNormal*sigma + mu
    }
}
