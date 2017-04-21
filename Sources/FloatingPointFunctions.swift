
import Accelerate

// MARK: - sqrt
public func sqrt(_ arg: NDArray) -> NDArray {
    return _sqrt(arg)
}

func _sqrt(_ arg: NDArray) -> NDArray {
    
    let volume = arg.volume
    var count = Int32(volume)
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = UnsafePointer(arg.data).advanced(by: arg.baseOffset)
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        vvsqrtf(dst, src, &count)
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    } else {
        let elements = gatherElements(arg)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        
        vvsqrtf(dst, elements, &count)
        return NDArray(shape: arg.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
