
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
        // Separate scattered major shape and strided minor shape
        let volume = array.volume
        let minorDims = stridedDims(shape: array.shape, strides: array.strides)
        let majorShape = [Int](array.shape.dropLast(minorDims))
        let majorStrides = [Int](array.strides.dropLast(minorDims))
        
        let stride = array.strides.last!
        let blockSize = array.shape.suffix(minorDims).prod()
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let offsets = getOffsets(shape: majorShape, strides: majorStrides)
        let _blockSize = vDSP_Length(blockSize)
        
        let src = array.startPointer
        var dstPtr = dst
        for offset in offsets {
            let src = src + offset
            
            vDSP_vclip(src, stride, &low, &high, dstPtr, 1, _blockSize)
            dstPtr += blockSize
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
