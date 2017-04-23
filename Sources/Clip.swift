
import Accelerate

public func clip(_ array: NDArray, low: Float, high: Float) -> NDArray {
    return min(max(array, low), high)
}

public func min(_ lhs: NDArray, _ rhs: Float) -> NDArray {
    return apply(lhs, rhs, vDSP_vmin)
}

public func min(_ lhs: Float, _ rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vDSP_vmin)
}

public func max(_ lhs: NDArray, _ rhs: Float) -> NDArray {
    return apply(lhs, rhs, vDSP_vmax)
}

public func max(_ lhs: Float, _ rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vDSP_vmax)
}


private typealias vDSP_func = (UnsafePointer<Float>, vDSP_Stride, UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

private func apply(_ array: NDArray, _ scalar: Float, _ vDSPfunc: vDSP_func) -> NDArray {
    
    var scalar = scalar
    
    if isDense(shape: array.shape, strides: array.strides) {
        let src = UnsafePointer(array.data).advanced(by: array.baseOffset)
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: array.data.count)
        defer { dst.deallocate(capacity: array.data.count) }
        vDSPfunc(src, 1, &scalar, 0, dst, 1, vDSP_Length(array.data.count))
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
        let numProc = ProcessInfo.processInfo.activeProcessorCount
        let offsetsBlockSize = Int(ceil(Float(offsets.count) / Float(numProc)))
        
        let _blockSize = vDSP_Length(blockSize)
        
        DispatchQueue.concurrentPerform(iterations: numProc) { i in
            var dstPtr = dst.advanced(by: i*offsetsBlockSize*blockSize)
            let start = i*offsetsBlockSize
            let end = start + min(offsetsBlockSize, offsets.count - i*offsetsBlockSize)
            
            guard start < end else { // can be empty
                return
            }
            for oi in start..<end {
                let offset = offsets[oi] + array.baseOffset
                let src = UnsafePointer(array.data).advanced(by: offset)
                
                vDSPfunc(src, stride, &scalar, 0, dstPtr, 1, _blockSize)
                dstPtr += blockSize
            }
        }
        
        return NDArray(shape: array.shape, elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
