
import Accelerate

/// Get minimum element.
public func min(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_minv)
}

/// Get maximum element.
public func max(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_maxv)
}

/// Calculate sum of all elements.
public func sum(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_sve)
}

/// Calculate mean of all elements.
public func mean(_ arg: NDArray) -> NDArray {
    return reduce(arg, vDSP_meanv)
}

/// Calculate mean and variance of all elements.
public func moments(_ arg: NDArray) -> (mean: NDArray, variance: NDArray) {
    let elements = gatherElements(arg)
    var sum: Float = 0
    var sum2: Float = 0
    vDSP_sve_svesq(UnsafePointer(elements), 1, &sum, &sum2, vDSP_Length(elements.count))
    let mean = sum / Float(elements.count)
    let mean2 = sum2 / Float(elements.count)
    return (NDArray(scalar: mean), NDArray(scalar: mean2 - mean*mean))
}

/// Calculate standard deviation of all elements.
public func variance(_ arg: NDArray) -> NDArray {
    return moments(arg).variance
}

/// Caluclate standard deviation of all elements.
public func stddev(_ arg: NDArray) -> NDArray {
    return sqrt(variance(arg))
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

/// Calculate mean and variance of elements along a given axis.
public func moments(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> (mean: NDArray, variance: NDArray) {
    let (mean, variance) = _moments(arg, along: axis)
    
    if keepDims {
        return (mean.expandDims(axis), variance.expandDims(axis))
    } else {
        return (mean, variance)
    }
}

/// Calculate variance of elements along a given axis.
public func variance(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    return moments(arg, along: axis, keepDims: keepDims).variance
}

/// Calculate standard deviations of elements along a given axis.
public func stddev(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    return sqrt(variance(arg, along: axis, keepDims: keepDims))
}

// MARK: Util
private func _moments(_ arg: NDArray, along axis: Int) -> (mean: NDArray, variance: NDArray) {
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
    let mean = NDArray(shape: newShape,
                       elements: [Float](UnsafeBufferPointer(start: dst1, count: volume)))
    
    vDSP_vsdiv(dst2, 1, &_count, dst2, 1, _volume)
    vDSP_vsq(dst1, 1, dst1, 1, _volume)
    vDSP_vsub(dst1, 1, dst2, 1, dst2, 1, _volume)
    
    let variance =  NDArray(shape: newShape,
                            elements: [Float](UnsafeBufferPointer(start: dst2, count: volume)))
    
    return (mean, variance)
}
