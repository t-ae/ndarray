
import Accelerate

/// Get minimum element.
public func min(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_minv)
}

/// Get maximum element.
public func max(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_maxv)
}

/// Caluclate sum of all elements.
public func sum(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_sve)
}

/// Caluclate mean of all elements.
public func mean(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_meanv)
}

/// Get minimal elements along a given axis.
public func min(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_minv)
}

/// Get maximum elements along a given axis.
public func max(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_maxv)
}

/// Get minimal elements' indices along a given axis.
public func argmin(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_minvi)
}

/// Get maximum elements' indices along a given axis.
public func argmax(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_maxvi)
}

/// Calculate sum of elements along a given axis.
public func sum(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_sve)
}

/// Calculate mean of elements along a given axis.
public func mean(_ arg: NDArray, along axis: Int) -> NDArray {
    return reduce(arg, along: axis, vDSP_meanv)
}

// MARK: Util
private typealias vDSP_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, vDSP_Length) -> Void
private typealias vDSP_index_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, UnsafeMutablePointer<vDSP_Length>, vDSP_Length) -> Void

private func reduce(_ arg: NDArray, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    let elements = gatherElements(arg)
    var result: Float = 0
    vDSPfunc(UnsafePointer(elements), 1, &result, vDSP_Length(elements.count))
    return NDArray(scalar: result)
}

private func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.prod()
    
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

private func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_index_reduce_func) -> NDArray {
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.prod()
    
    let dst = UnsafeMutablePointer<vDSP_Length>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    var e: Float = 0
    
    let offsets = getOffsets(shape: newShape, strides: arg.strides.removing(at: axis))
    let count = vDSP_Length(arg.shape[axis])
    let stride = arg.strides[axis]
    
    var dstPtr = dst
    for offset in offsets {
        let src = UnsafePointer(arg.data).advanced(by: offset + arg.baseOffset)
        vDSPfunc(src, stride, &e, dstPtr, count)
        dstPtr += 1
    }
    
    // all indices are multiplied with stride.
    let indices = UnsafeBufferPointer<vDSP_Length>(start: dst, count: volume)
    return NDArray(shape: newShape,
                   elements: indices.map { Float(Int($0)/stride) })
}
