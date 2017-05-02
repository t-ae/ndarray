
import Accelerate

/// Get minimums for each pair elements
public func minimum(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_vmin)
}

/// Get maximums for each pair elements
public func maximum(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_vmax)
}

extension NDArray {
    /// Clip lower values.
    public func clipped(low: Float) -> NDArray {
        return clip(self, low: low, high: Float.greatestFiniteMagnitude)
    }
    
    /// Clip higher values.
    public func clipped(high: Float) -> NDArray {
        return clip(self, low: -Float.greatestFiniteMagnitude, high: high)
    }
    
    /// Clip lower and higher values.
    public func clipped(low: Float, high: Float) -> NDArray {
        return clip(self, low: low, high: high)
    }
}

// MARK: Util
private typealias vDSP_func = (UnsafePointer<Float>, vDSP_Stride, UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

private func clip(_ array: NDArray, low: Float, high: Float) -> NDArray {
    
    var low = low
    var high = high
    
    if isDense(shape: array.shape, strides: array.strides) {
        let src = array.startPointer
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: array.data.count)
        defer { dst.deallocate(capacity: array.data.count) }
        vDSP_vclip(src, 1, &low, &high, dst, 1, vDSP_Length(array.data.count))
        return NDArray(shape: array.shape,
                       strides: array.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: array.data.count)))
    } else {
        let ndim = array.shape.count
        let volume = array.shape.prod()
        
        let axis = getLeastStrideAxis(array.strides)
        let srcStride = array.strides[axis]
        let dims = getStridedDims(shape: array.shape, strides: array.strides, from: axis)
        
        let outerShape = [Int](array.shape[0..<axis-dims+1] + array.shape[axis+1..<ndim])
        let outerStrides = [Int](array.strides[0..<axis-dims+1] + array.strides[axis+1..<ndim])
        let blockSize = array.shape[axis-dims+1...axis].prod()
        
        let dstStrides = contiguousStrides(shape: array.shape)
        let dstOuterStrides = [Int](dstStrides[0..<axis-dims+1] + dstStrides[axis+1..<ndim])
        
        let dstStride = dstStrides[axis]
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let srcOffsets = getOffsets(shape: outerShape, strides: outerStrides)
        let dstOffsets = getOffsets(shape: outerShape, strides: dstOuterStrides)
        let _blockSize = vDSP_Length(blockSize)
        
        let src = array.startPointer
        for (os, od) in zip(srcOffsets, dstOffsets) {
            let src = src + os
            let dst = dst + od
            
            vDSP_vclip(src, srcStride, &low, &high, dst, dstStride, _blockSize)
        }
        
        return NDArray(shape: array.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}

private func apply(_ lhs: NDArray, _ rhs: NDArray, _ vDSPfunc: vDSP_func) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    let strDims = min(stridedDims(shape: lhs.shape, strides: lhs.strides),
                      stridedDims(shape: rhs.shape, strides: rhs.strides))
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
    let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
    
    let lStride = vDSP_Stride(lhs.strides.last!)
    let rStride = vDSP_Stride(rhs.strides.last!)
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
