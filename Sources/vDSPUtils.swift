
import Accelerate

// MARK: - Binary
// MARK: Vector-Scalar operation
typealias vDSP_vs_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafePointer<Float>,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ lhs: NDArray,
           _ rhs: Float,
           _ vDSPfunc: vDSP_vs_func) -> NDArray {
    
    let strDims = stridedDims(shape: lhs.shape, strides: lhs.strides)
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let minorShape = lhs.shape.suffix(strDims)
    
    let blockSize = minorShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: lhs.volume)
    defer { dst.deallocate(capacity: lhs.volume) }
    
    let majorStrides = [Int](lhs.strides.dropLast(strDims))
    let offsets = getOffsets(shape: majorShape, strides: majorStrides)
    
    var rhs = rhs
    let stride = vDSP_Stride(lhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let src = lhs.startPointer
    var dstPtr = dst
    for offset in offsets {
        let src = src + offset
        vDSPfunc(src, stride,
                 &rhs,
                 dstPtr, 1, _blockSize)
        
        dstPtr += blockSize
    }
    
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: lhs.volume)))
}

// MARK: Scalar-Vector operation
typealias vDSP_sv_func = (UnsafePointer<Float>,
    UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ lhs: Float,
           _ rhs: NDArray,
           _ vDSPfunc: vDSP_sv_func) -> NDArray {
    
    let strDims = stridedDims(shape: rhs.shape, strides: rhs.strides)
    
    let majorShape = [Int](rhs.shape.dropLast(strDims))
    let minorShape = rhs.shape.suffix(strDims)
    
    let blockSize = minorShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: rhs.volume)
    defer { dst.deallocate(capacity: rhs.volume) }
    
    let majorStrides = [Int](rhs.strides.dropLast(strDims))
    let offsets = getOffsets(shape: majorShape, strides: majorStrides)
    
    var lhs = lhs
    let stride = vDSP_Stride(rhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let src = rhs.startPointer
    var dstPtr = dst
    for offset in offsets {
        let src = src + offset
        vDSPfunc(&lhs,
                 src, stride,
                 dstPtr, 1, _blockSize)
        
        dstPtr += blockSize
    }
    
    return NDArray(shape: rhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: rhs.volume)))
}

// MARK: Vector-Vector operation
typealias vDSP_vv_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ lhs: NDArray, _ rhs: NDArray, _ vDSPfunc: vDSP_vv_func) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    let strDims = min(stridedDims(shape: lhs.shape, strides: lhs.strides),
                      stridedDims(shape: rhs.shape, strides: rhs.strides))
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
    let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
    
    let lStride = vDSP_Stride(lhs.strides.last ?? 0)
    let rStride = vDSP_Stride(rhs.strides.last ?? 0)
    let blockSize = lhs.shape.suffix(strDims).prod()
    let _blockSize = vDSP_Length(blockSize)
    
    let lOffsets = getOffsets(shape: majorShape, strides: lMajorStrides)
    let rOffsets = getOffsets(shape: majorShape, strides: rMajorStrides)
    
    
    let lSrc = lhs.startPointer
    let rSrc = rhs.startPointer
    var dstPtr = dst
    for (lo, ro) in zip(lOffsets, rOffsets) {
        vDSPfunc(lSrc + lo, lStride,
                 rSrc + ro, rStride,
                 dstPtr, 1, _blockSize)
        dstPtr += blockSize
    }
    
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
}

// MARK: - Reduce

typealias vDSP_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, vDSP_Length) -> Void

// Reduce all elements.
func reduce(_ arg: NDArray, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    let elements = gatherElements(arg)
    var result: Float = 0
    vDSPfunc(UnsafePointer(elements), 1, &result, vDSP_Length(elements.count))
    return NDArray(scalar: result)
}

/// Reduce along a given axis.
func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    let offsets = getOffsets(shape: newShape, strides: arg.strides.removing(at: axis))
    let count = vDSP_Length(arg.shape[axis])
    let stride = arg.strides[axis]
    
    let src = arg.startPointer
    var dstPtr = dst
    for offset in offsets {
        let src = src + offset
        vDSPfunc(src, stride, dstPtr, count)
        dstPtr += 1
    }
    
    return NDArray(shape: newShape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
}

typealias vDSP_index_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, UnsafeMutablePointer<vDSP_Length>, vDSP_Length) -> Void

// Reduce along a given axis (for argmin, argmux).
func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_index_reduce_func) -> NDArray {
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
    let src = arg.startPointer
    for offset in offsets {
        let src = src + offset
        vDSPfunc(src, stride, &e, dstPtr, count)
        dstPtr += 1
    }
    
    // all indices are multiplied with stride.
    let indices = UnsafeBufferPointer<vDSP_Length>(start: dst, count: volume)
    return NDArray(shape: newShape,
                   elements: indices.map { Float(Int($0)/stride) })
}
