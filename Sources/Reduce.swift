
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

/// Caluclate standard deviation of all elements.
public func std(_ arg: NDArray) -> NDArray {
    let elements = gatherElements(arg)
    var sum: Float = 0
    var sum2: Float = 0
    vDSP_sve_svesq(UnsafePointer(elements), 1, &sum, &sum2, vDSP_Length(elements.count))
    let mean = sum / Float(elements.count)
    let mean2 = sum2 / Float(elements.count)
    return NDArray(scalar: sqrtf(mean2 - mean*mean))
}

/// Get minimal elements along a given axis.
public func min(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = reduce(arg, along: axis, vDSP_minv)
    return keepDims ? ret.expandDims(axis) : ret
}

/// Get maximum elements along a given axis.
public func max(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = reduce(arg, along: axis, vDSP_maxv)
    
    return keepDims ? ret.expandDims(axis) : ret
}

/// Get minimal elements' indices along a given axis.
public func argmin(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = reduce(arg, along: axis, vDSP_minvi)
    
    return keepDims ? ret.expandDims(axis) : ret
}

/// Get maximum elements' indices along a given axis.
public func argmax(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = reduce(arg, along: axis, vDSP_maxvi)
    
    return keepDims ? ret.expandDims(axis) : ret
}

/// Calculate sum of elements along a given axis.
public func sum(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = reduce(arg, along: axis, vDSP_sve)
    
    return keepDims ? ret.expandDims(axis) : ret
}

/// Calculate mean of elements along a given axis.
public func mean(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = reduce(arg, along: axis, vDSP_meanv)
    
    return keepDims ? ret.expandDims(axis) : ret
}

/// Calculate standard deviations of elements along a given axis.
public func std(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = _std(arg, along: axis)
    
    return keepDims ? ret.expandDims(axis) : ret
}

// MARK: Util
private func _std(_ arg: NDArray, along axis: Int) -> NDArray {
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.prod()
    
    let dst1 = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    let dst2 = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer {
        dst1.deallocate(capacity: volume)
        dst2.deallocate(capacity: volume)
    }
    
    let offsets = OffsetSequence(shape: newShape, strides: arg.strides.removing(at: axis))
    let count = vDSP_Length(arg.shape[axis])
    let stride = arg.strides[axis]
    
    let src = arg.startPointer
    var dst1Ptr = dst1
    var dst2Ptr = dst2
    for offset in offsets {
        let src = src + offset
        vDSP_sve_svesq(src, stride, dst1Ptr, dst2Ptr, count)
        dst1Ptr += 1
        dst2Ptr += 1
    }
    var _count = Float(count)
    let _volume = vDSP_Length(volume)
    vDSP_vsdiv(dst1, 1, &_count, dst1, 1, _volume)
    vDSP_vsdiv(dst2, 1, &_count, dst2, 1, _volume)
    vDSP_vsq(dst1, 1, dst1, 1, _volume)
    vDSP_vsub(dst1, 1, dst2, 1, dst2, 1, _volume)
    var _volume2 = Int32(volume)
    vvsqrtf(dst2, dst2, &_volume2)

    return NDArray(shape: newShape,
                   elements: [Float](UnsafeBufferPointer(start: dst2, count: volume)))
}
