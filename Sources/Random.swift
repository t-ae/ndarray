
import Foundation
import Accelerate

extension NDArray {
    /// Create randomly initialized NDArray.
    ///
    /// All values are sampled from the interval [`low`, `high`).
    public static func uniform(low: Float = 0, high: Float = 1, shape: [Int]) -> NDArray {
        precondition(shape.all { $0 >= 0 })
        let size = shape.prod()
        let dst = UnsafeMutablePointer<UInt32>.allocate(capacity: size)
        let dst2 = UnsafeMutablePointer<Float>.allocate(capacity: size)
        defer {
            dst.deallocate(capacity: size)
            dst2.deallocate(capacity: size)
        }
        arc4random_buf(dst, MemoryLayout<UInt32>.size * size)
        
        vDSP_vfltu32(dst, 1, dst2, 1, vDSP_Length(size))
        
        let array = NDArray(shape: shape,
                            elements: [Float](UnsafeBufferPointer(start: dst2, count: size)))
        
        return (high - low) * array / Float(UInt32.max) + low
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
