
import Accelerate

public func floor(_ arg: NDArray) -> NDArray {
    return apply(arg, vvfloorf)
}

public func ceil(_ arg: NDArray) -> NDArray {
    return apply(arg, vvceilf)
}

public func round(_ arg: NDArray) -> NDArray {
    return apply(arg, vvnintf)
}

public func sqrt(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsqrtf)
}

public func log(_ arg: NDArray) -> NDArray {
    return apply(arg, vvlogf)
}

public func log2(_ arg: NDArray) -> NDArray {
    return apply(arg, vvlog2f)
}

public func log10(_ arg: NDArray) -> NDArray {
    return apply(arg, vvlog10f)
}

public func exp(_ arg: NDArray) -> NDArray {
    return apply(arg, vvexpf)
}

public func exp2(_ arg: NDArray) -> NDArray {
    return apply(arg, vvexp2f)
}

public func sin(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsinf)
}

public func cos(_ arg: NDArray) -> NDArray {
    return apply(arg, vvcosf)
}

public func tan(_ arg: NDArray) -> NDArray {
    return apply(arg, vvtanf)
}

public func asin(_ arg: NDArray) -> NDArray {
    return apply(arg, vvasinf)
}

public func acos(_ arg: NDArray) -> NDArray {
    return apply(arg, vvacosf)
}

public func atan(_ arg: NDArray) -> NDArray {
    return apply(arg, vvatanf)
}

public func sinh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsinhf)
}

public func cosh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvcoshf)
}

public func tanh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvtanhf)
}

// MARK: Util
private typealias vvUnaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

private func apply(_ arg: NDArray, _ vvfunc: vvUnaryFunc) -> NDArray {
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        var count = Int32(arg.data.count)
        let src = UnsafePointer(arg.data).advanced(by: arg.baseOffset)
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
        defer { dst.deallocate(capacity: arg.data.count) }
        vvfunc(dst, src, &count)
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: arg.data.count)))
    } else if arg.strides.last == 1 {
        let volume = arg.volume
        let strDims = stridedDims(shape: arg.shape, strides: arg.strides)
        
        let majorShape = [Int](arg.shape.dropLast(strDims))
        let majorStrides = [Int](arg.strides.dropLast(strDims))
        let blockSize = arg.shape.suffix(strDims).reduce(1, *)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let offsets = getOffsets(shape: majorShape, strides: majorStrides)
        let numProc = ProcessInfo.processInfo.activeProcessorCount
        let offsetsBlockSize = Int(ceil(Float(offsets.count) / Float(numProc)))
        
        var _blockSize = Int32(blockSize)
        
        DispatchQueue.concurrentPerform(iterations: numProc) { i in
            var dstPtr = dst.advanced(by: i*offsetsBlockSize*blockSize)
            let start = i * offsetsBlockSize
            let end = start + min(offsetsBlockSize, offsets.count - i*offsetsBlockSize)
            
            guard start < end else { // can be empty
                return
            }
            for oi in start..<end {
                let offset = offsets[oi] + arg.baseOffset
                let src = UnsafePointer(arg.data).advanced(by: offset)
                
                vvfunc(dstPtr, src, &_blockSize)
                dstPtr += blockSize
            }
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
