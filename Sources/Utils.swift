
import Accelerate

/// Get continuous strides
func continuousStrides(shape: [Int]) -> [Int] {
    guard !shape.isEmpty else {
        return []
    }
    var strides = [1]
    for s in shape.dropFirst().reversed() {
        strides.insert(strides[0]*s, at: 0)
    }
    return strides
}

/// Get offset
func indexOffset(strides: [Int], ndIndex: [Int]) -> Int {
    precondition(strides.count == ndIndex.count)
    return zip(ndIndex, strides)
        .map(*)
        .reduce(0, +)
}

/// Calculate how many dims are continuous
func continuousDims(shape: [Int], strides: [Int]) -> Int {
    precondition(shape.count == strides.count)
    var continuousDims = 0
    var stride = 1
    for (s, str) in zip(shape.reversed(), strides.reversed()) {
        if stride == str {
            continuousDims += 1
        } else {
            break
        }
        stride *= s
    }
    return continuousDims
}

/// Calculate how many dims are strided
func stridedDims(shape: [Int], strides: [Int]) -> Int {
    precondition(shape.count == strides.count)
    var stridedDims = 0
    guard var stride = strides.last else {
        return 0
    }
    for (s, str) in zip(shape.reversed(), strides.reversed()) {
        if stride == str {
            stridedDims += 1
        } else {
            break
        }
        stride *= s
    }
    return stridedDims
}

/// Gather elements
func gatherElements(_ arg: NDArray, forceUniqueReference: Bool = false) -> [Float] {
    
    let volume = arg.volume
    
    if arg.isContinuous {
        if arg.baseOffset == 0 && volume == arg.data.count {
            if forceUniqueReference {
                let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
                defer { dst.deallocate(capacity: arg.data.count) }
                memcpy(dst, arg.data, arg.data.count*MemoryLayout<Float>.size)
                return Array(UnsafeBufferPointer(start: dst, count: arg.data.count))
            } else {
                return arg.data
            }
        } else {
            let start = arg.baseOffset
            let end = start + volume
            return Array(arg.data[start..<end])
        }
    } else {
        let minorDims = stridedDims(shape: arg.shape, strides: arg.strides)
        let majorShape = [Int](arg.shape.dropLast(minorDims))
        let majorStrides = [Int](arg.strides.dropLast(minorDims))
        let minorZeros = [Int](repeating: 0, count: minorDims)
        
        let stride = Int32(arg.strides.last!)
        let count = arg.shape.suffix(minorDims).reduce(1, *)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        var dstPtr = dst
        for majorIndex in NDIndexSequence(shape: majorShape) {
            let offset = indexOffset(strides: majorStrides, ndIndex: majorIndex) + arg.baseOffset
            let src = UnsafePointer(arg.data) + offset
            
            cblas_scopy(Int32(count), src, stride, dstPtr, 1)
            dstPtr += count
        }
        return [Float](UnsafeBufferPointer(start: dst, count: volume))
    }
}

/// Broadcast two arrays
func broadcast(_ lhs: NDArray, _ rhs: NDArray) -> (NDArray, NDArray) {
    if lhs.shape == rhs.shape {
        return (lhs, rhs)
    }
    
    var (lShape, rShape) = (lhs.shape, rhs.shape)
    var (lStrides, rStrides) = (lhs.strides, rhs.strides)
    
    let d = lShape.count - rShape.count
    if d < 0 {
        lShape = [Int](repeating: 1, count: -d) + lShape
        lStrides = [Int](repeating: 0, count: -d) + lStrides
    } else if(d > 0) {
        rShape = [Int](repeating: 1, count: d) + rShape
        rStrides = [Int](repeating: 0, count: d) + rStrides
    }
    
    for i in 0..<lShape.count {
        if lShape[i] == rShape[i] {
            continue
        } else if(lShape[i] == 1) {
            lShape[i] = rShape[i]
            lStrides[i] = 0
        } else if(rShape[i] == 1) {
            rShape[i] = lShape[i]
            rStrides[i] = 0
        } else {
            preconditionFailure()
        }
    }
    
    let lArray = NDArray(shape: lShape, strides: lStrides, baseOffset: lhs.baseOffset, data: lhs.data)
    let rArray = NDArray(shape: rShape, strides: rStrides, baseOffset: rhs.baseOffset, data: rhs.data)
    
    return (lArray, rArray)
}

/// Broadcast arg to shape
func broadcast(_ arg: NDArray, to shape: [Int]) -> NDArray {
    precondition(arg.shape.count <= shape.count)
    if arg.shape == shape {
        return arg
    }
    
    let d = shape.count - arg.shape.count
    var newShape = [Int](repeating: 1, count: d) + arg.shape
    var newStrides = [Int](repeating: 0, count: d) + arg.strides
    
    for i in 0..<newShape.count {
        if newShape[i] == shape[i] {
            continue
        } else if newShape[i] == 1 {
            newShape[i] = shape[i]
            newStrides[i] = 0
        } else {
            preconditionFailure()
        }
    }
    
    return NDArray(shape: newShape, strides: newStrides, baseOffset: arg.baseOffset, data: arg.data)
    
}

/// Return normalized index
/// - Check all numbers in valid range
/// - Process minus number
func normalizeIndex(shape: [Int], ndIndex: [Int]) -> [Int] {
    precondition(shape.count == ndIndex.count)
    
    var ndIndex = ndIndex
    for i in 0..<ndIndex.count {
        if ndIndex[i] < -shape[i] || ndIndex[i] >= shape[i] {
            preconditionFailure()
        }
        if ndIndex[i] < 0 {
            ndIndex[i] += shape[i]
        }
    }
    return ndIndex
}

extension Array where Element == Bool {
    func all() -> Bool {
        for e in self {
            if !e {
                return false
            }
        }
        return true
    }
    
    func some() -> Bool {
        for e in self {
            if e {
                return true
            }
        }
        return false
    }
}

extension Array {
    func removed(at index: Int) -> Array {
        var ret = self
        ret.remove(at: index)
        return ret
    }
}
