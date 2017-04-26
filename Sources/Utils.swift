
import Accelerate

/// Check if elements are aligned continuously.
func isContinuous(shape: [Int], strides: [Int]) -> Bool {
    return shape.isEmpty || (strides.last == 1 && isStrided(shape: shape, strides: strides))
}

/// Check if whole elements are strided.
func isStrided(shape: [Int], strides: [Int]) -> Bool {
    return shape.count == stridedDims(shape: shape, strides: strides)
}

/// Check if elements are densely placed.
///
/// Doesn't permit minus strides.
func isDense(shape: [Int], strides: [Int]) -> Bool {
    
    if shape.count == 0 {
        return true
    }
    
    let nonZeros = strides.filter { $0 != 0 }
    let numZeros = strides.count - nonZeros.count
    
    var strideCount = 0
    var stride = 1
    while true {
        if let index = strides.index(of: stride) {
            stride *= shape[index]
            strideCount += 1
        } else {
            return strideCount + numZeros == shape.count
        }
    }
}

/// Get continuous strides.
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

/// Get offset.
func indexOffset(strides: [Int], ndIndex: [Int]) -> Int {
    precondition(strides.count == ndIndex.count)
    return zip(ndIndex, strides)
        .map(*)
        .sum()
}

/// Get indices in row major order.
func getIndices(shape: [Int]) -> [[Int]] {
    guard !shape.isEmpty else {
        return [[]]
    }
    guard !shape.contains(0) else {
        return []
    }
    
    var index = [Int](repeating: 0, count: shape.count)
    var indices: [[Int]] = []
    let last = index.count - 1
    
    repeat {
        indices.append(index)
        index[last] += 1
        for i in 0..<last {
            guard index[last-i] >= shape[last-i] else {
                break
            }
            index[last-i] = 0
            index[last-i-1] += 1
        }
    } while index[0] != shape[0]
    
    return indices
}

/// Get offsets in row major order.
func getOffsets(shape: [Int], strides: [Int]) -> [Int] {
    precondition(shape.count == strides.count)
    guard !shape.isEmpty else {
        return [0]
    }
    guard !shape.contains(0) else {
        return []
    }
    
    var index = [Int](repeating: 0, count: shape.count)
    var offset = 0
    
    let volume = shape.prod()
    let dst = UnsafeMutablePointer<Int>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    let last = index.count - 1
    
    var dstPtr = dst
    repeat {
        dstPtr.pointee = offset
        dstPtr += 1
        index[last] += 1
        offset += strides[last]
        
        for i in 0..<last {
            guard index[last-i] >= shape[last-i] else {
                break
            }
            index[last-i] = 0
            offset -= strides[last-i]*shape[last-i]
            index[last-i-1] += 1
            offset += strides[last-i-1]
        }
    } while index[0] != shape[0]
    
    return [Int](UnsafeBufferPointer(start: dst, count: volume))
}

/// Calculate how many dims are strided.
func stridedDims(shape: [Int], strides: [Int]) -> Int {
    precondition(shape.count == strides.count)
    var stridedDims = 0
    guard var stride = strides.last else {
        return 0
    }
    for (s, str) in zip(shape.reversed(), strides.reversed()) {
        if s == 1 {
            stridedDims += 1
        }else if stride == str {
            stridedDims += 1
            stride *= s
        } else {
            break
        }
    }
    return stridedDims
}

/// Calculate how many dims are dense.
func denseDims(shape: [Int], strides: [Int]) -> Int {
    precondition(shape.count == strides.count)
    
    var contStr = continuousStrides(shape: shape)[0..<strides.count]
    var strides = strides[0..<strides.count]
    for i in 0..<shape.count {
        if Set(strides) == Set(contStr) {
            return shape.count-i
        }
        strides = strides.dropFirst()
        contStr = contStr.dropFirst()
    }
    return 1
}

/// Gather elements.
func gatherElements(_ arg: NDArray, forceUniqueReference: Bool = false) -> [Float] {
    
    let volume = arg.volume
    
    if isContinuous(shape: arg.shape, strides: arg.strides) {
        if volume == arg.data.count {
            if forceUniqueReference {
                let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
                defer { dst.deallocate(capacity: arg.data.count) }
                cblas_scopy(Int32(arg.data.count), arg.data, 1, dst, 1)
                // memcpy(dst, arg.data, arg.data.count*MemoryLayout<Float>.size)
                return [Float](UnsafeBufferPointer(start: dst, count: arg.data.count))
            } else {
                return arg.data
            }
        } else {
            let start = arg.baseOffset
            let end = start + volume
            return [Float](arg.data[start..<end])
        }
    } else {
        // Separate scattered major shape and strided minor shape
        let minorDims = stridedDims(shape: arg.shape, strides: arg.strides)
        let majorShape = [Int](arg.shape.dropLast(minorDims))
        let majorStrides = [Int](arg.strides.dropLast(minorDims))
        
        let stride = Int32(arg.strides.last!)
        let blockSize = arg.shape.suffix(minorDims).prod()
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let offsets = getOffsets(shape: majorShape, strides: majorStrides)
        let _blockSize = Int32(blockSize)
        
        let src = arg.startPointer
        var dstPtr = dst
        for offset in offsets {
            let src = src + offset
            
            cblas_scopy(_blockSize, src, stride, dstPtr, 1)
            dstPtr += blockSize
        }
        
        return [Float](UnsafeBufferPointer(start: dst, count: volume))
    }
}

/// Return normalized index.
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

/// Return normalized axis.
/// - Check axis is in valid range
/// - Process minus number
func normalizeAxis(axis: Int, ndim: Int) -> Int {
    var axis = axis
    if axis < 0 {
        axis += ndim
    }
    precondition(axis >= 0 && axis < ndim)
    return axis
}

extension NDArray {
    var startPointer: UnsafePointer<Float> {
        return UnsafePointer(data) + baseOffset
    }
}

extension Array {
    func all(cond: (Element)->Bool) -> Bool {
        for e in self {
            if !cond(e) {
                return false
            }
        }
        return true
    }
    
    func removing(at index: Int) -> Array {
        var ret = self
        ret.remove(at: index)
        return ret
    }
    
    func inserting(_ newElement: Element, at: Int) -> Array {
        var ret = self
        ret.insert(newElement, at: at)
        return ret
    }
}

extension Sequence where Iterator.Element == Int {
    func sum() -> Int {
        var ret = 0
        for e in self {
            ret += e
        }
        return ret
    }
    
    func prod() -> Int {
        var ret = 1
        for e in self {
            ret *= e
        }
        return ret
    }
}
