import Accelerate

precedencegroup PowPrecedense {
    associativity: left
    higherThan: MultiplicationPrecedence
}

infix operator **: PowPrecedense

public func **(lhs: NDArray, rhs: Float) -> NDArray {
    return lhs ** NDArray(scalar: rhs)
}

public func **(lhs: Float, rhs: NDArray) -> NDArray {
    return NDArray(scalar: lhs) ** rhs
}

public func **(lhs: NDArray, rhs: NDArray) -> NDArray {
    
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    if lhs.strides.last == 1 && rhs.strides.last == 1 {
        let strDims = min(stridedDims(shape: lhs.shape, strides: lhs.strides),
                          stridedDims(shape: rhs.shape, strides: rhs.strides))
        
        let majorShape = [Int](lhs.shape.dropLast(strDims))
        let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
        let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
        let blockSize = lhs.shape.suffix(strDims).prod()
        
        let lOffsets = getOffsets(shape: majorShape, strides: lMajorStrides)
        let rOffsets = getOffsets(shape: majorShape, strides: rMajorStrides)
        var _blockSize = Int32(blockSize)
        
        let lSrc = lhs.startPointer
        let rSrc = rhs.startPointer
        var dstPtr = dst
        for (lo, ro) in zip(lOffsets, rOffsets) {
            vvpowf(dstPtr, rSrc + ro, lSrc + lo, &_blockSize)
            dstPtr += blockSize
        }
        
        return NDArray(shape: lhs.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    } else {
        var _volume = Int32(volume)
        
        let lElements = gatherElements(lhs)
        let rElements = gatherElements(rhs)
        
        vvpowf(dst, rElements, lElements, &_volume)
        
        return NDArray(shape: lhs.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
