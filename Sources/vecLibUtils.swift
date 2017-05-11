
import Accelerate

typealias vvUnaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

func apply(_ arg: NDArray, _ vvfunc: vvUnaryFunc) -> NDArray {
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        var count = Int32(arg.data.count)
        let src = arg.startPointer
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
        defer { dst.deallocate(capacity: arg.data.count) }
        vvfunc(dst, src, &count)
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: arg.data.count)))
    } else if arg.strides.last == 1 {
        let volume = arg.volume
        let strDims = getStridedDims(shape: arg.shape, strides: arg.strides)
        
        let majorShape = [Int](arg.shape.dropLast(strDims))
        let majorStrides = [Int](arg.strides.dropLast(strDims))
        let blockSize = arg.shape.suffix(strDims).prod()
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let offsets = OffsetSequence(shape: majorShape, strides: majorStrides)
        var _blockSize = Int32(blockSize)
        
        let src = arg.startPointer
        var dstPtr = dst
        for offset in offsets {
            let src = src + offset
            
            vvfunc(dstPtr, src, &_blockSize)
            dstPtr += blockSize
        }
        
        return NDArray(shape: arg.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
        
    } else {
        let volume = arg.volume
        var count = Int32(volume)
        let elements = gatherElements(arg)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        
        vvfunc(dst, elements, &count)
        return NDArray(shape: arg.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}

typealias vvBinaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

func apply(_ lhs: NDArray, _ rhs: NDArray, _ vvfunc: vvBinaryFunc) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    if lhs.strides.last == 1 && rhs.strides.last == 1 {
        let strDims = min(getStridedDims(shape: lhs.shape, strides: lhs.strides),
                          getStridedDims(shape: rhs.shape, strides: rhs.strides))
        
        let majorShape = [Int](lhs.shape.dropLast(strDims))
        let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
        let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
        let blockSize = lhs.shape.suffix(strDims).prod()
        
        let lOffsets = OffsetSequence(shape: majorShape, strides: lMajorStrides)
        let rOffsets = OffsetSequence(shape: majorShape, strides: rMajorStrides)
        var _blockSize = Int32(blockSize)
        
        let lSrc = lhs.startPointer
        let rSrc = rhs.startPointer
        var dstPtr = dst
        for (lo, ro) in zip(lOffsets, rOffsets) {
            vvfunc(dstPtr, lSrc + lo, rSrc + ro, &_blockSize)
            dstPtr += blockSize
        }
        
        return NDArray(shape: lhs.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    } else {
        var _volume = Int32(volume)
        
        let lElements = gatherElements(lhs)
        let rElements = gatherElements(rhs)
        
        vvfunc(dst, lElements, rElements, &_volume)
        
        return NDArray(shape: lhs.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
