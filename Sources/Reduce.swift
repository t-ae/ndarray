
import Accelerate

public func min(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_minv)
}

public func max(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_maxv)
}

public func sum(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_sve)
}

public func mean(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_meanv)
}

// MARK: Util
private typealias vDSP_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, vDSP_Length) -> Void

private func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    
    let axis = normalizeAxis(axis: axis, ndim: arg.shape.count)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.reduce(1, *)
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    let offsets = getOffsets(shape: newShape, strides: arg.strides.removing(at: axis))
    let count = vDSP_Length(arg.shape[axis])
    let stride = arg.strides[axis]
    
    var dstPtr = dst
    for offset in offsets {
        let src = UnsafePointer(arg.data).advanced(by: offset + arg.baseOffset)
        vDSPfunc(src, stride, dstPtr, count)
        dstPtr += 1
    }
    
    return NDArray(shape: newShape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
}