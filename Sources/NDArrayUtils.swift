
import Accelerate

/// Check if elements are aligned contiguously.
func isContiguous(shape: [Int], strides: [Int]) -> Bool {
    assert(shape.count == strides.count)
    return shape.isEmpty || (strides.last == 1 && shape.count == getStridedDims(shape: shape, strides: strides))
}

/// Check if elements are densely placed.
///
/// Doesn't permit minus strides.
func isDense(shape: [Int], strides: [Int]) -> Bool {
    assert(shape.count == strides.count)
    
    if shape.count == 0 {
        return true
    }
    
    let nonZeros = strides.filter { $0 != 0 }
    let numZeros = strides.count - nonZeros.count
    
    var strideCount = 0
    var stride = 1
    while true {
        if strideCount == shape.count {
            return true
        } else if let index = strides.index(of: stride) {
            stride *= shape[index]
            strideCount += 1
        } else {
            return strideCount + numZeros == shape.count
        }
    }
}

/// Get contiguous strides.
func getContiguousStrides(shape: [Int]) -> [Int] {
    assert(shape.all { $0 >= 0 })
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
func getIndexOffset(strides: [Int], ndIndex: [Int]) -> Int {
    assert(strides.count == ndIndex.count)
    assert(ndIndex.all { $0 >= 0 })
    return zip(ndIndex, strides)
        .map(*)
        .sum()
}

/// Get indices in row major order.
func getIndices(shape: [Int]) -> [[Int]] {
    assert(shape.all { $0 >= 0 })
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
    assert(shape.count == strides.count)
    assert(shape.all { $0 >= 0 })
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
func getStridedDims(shape: [Int], strides: [Int]) -> Int {
    assert(shape.count == strides.count)
    assert(shape.all { $0 >= 0 })
    var stridedDims = 0
    guard var stride = strides.last else {
        return 0
    }
    var strides = strides
    if stride < 0 {
        stride = -stride
        strides = strides.map { -$0 }
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

/// Get the axis which has least stride.
func getLeastStrideAxis(_ strides: [Int]) -> Int {
    assert(!strides.isEmpty)
    var axis = 0
    var minimum = abs(strides[0])
    for (i, s) in strides.enumerated().dropFirst() {
        let sa = abs(s)
        if sa < minimum {
            minimum = sa
            axis = i
        }
    }
    return axis
}

/// Calculate how many dimensions are strided from axis.
func getStridedDims(shape: [Int], strides: [Int], from axis: Int) -> Int {
    assert(shape.count == strides.count)
    assert(shape.all { $0 >= 0 })
    assert(0 <= axis && axis < shape.count)
    
    var stride = strides[axis]
    var stridedDims = 0
    
    var strides = strides
    if stride < 0 {
        stride = -stride
        strides = strides.map { -$0 }
    }
    for i in (0...axis).reversed() {
        if shape[i] == 1 {
            stridedDims += 1
        } else if strides[i] == stride {
            stridedDims += 1
            stride *= shape[i]
        } else {
            break
        }
    }
    
    return stridedDims
}

/// Gather elements.
func gatherElements(_ arg: NDArray, forceUniqueReference: Bool = false) -> [Float] {
    
    let volume = arg.volume
    let ndim = arg.ndim
    
    if isContiguous(shape: arg.shape, strides: arg.strides) {
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

        let axis = getLeastStrideAxis(arg.strides)
        let srcStride = Int32(arg.strides[axis])
        let dims = getStridedDims(shape: arg.shape, strides: arg.strides, from: axis)
    
        let outerShape = [Int](arg.shape[0..<axis-dims+1] + arg.shape[axis+1..<ndim])
        let outerStrides = [Int](arg.strides[0..<axis-dims+1] + arg.strides[axis+1..<ndim])
        let blockSize = arg.shape[axis-dims+1...axis].prod()
        
        let dstStrides = getContiguousStrides(shape: arg.shape)
        let dstOuterStrides = [Int](dstStrides[0..<axis-dims+1] + dstStrides[axis+1..<ndim])
        
        let dstStride = Int32(dstStrides[axis])
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let srcOffsets = getOffsets(shape: outerShape, strides: outerStrides)
        let dstOffsets = getOffsets(shape: outerShape, strides: dstOuterStrides)
        let _blockSize = Int32(blockSize)
        
        let src: UnsafePointer<Float>
        if srcStride < 0 {
            src = arg.startPointer + (blockSize-1) * Int(srcStride)
        } else {
            src = arg.startPointer
        }
        for (os, od) in zip(srcOffsets, dstOffsets) {
            let src = src + os
            let dst = dst + od
            
            cblas_scopy(_blockSize, src, srcStride, dst, dstStride)
        }
        
        return [Float](UnsafeBufferPointer(start: dst, count: volume))
    }
}

/// Return normalized index.
/// - Check all numbers in valid range
/// - Process minus number
func normalizeIndex(shape: [Int], ndIndex: [Int]) -> [Int] {
    assert(shape.count == ndIndex.count)
    assert(shape.all { $0 >= 0 })
    
    var ndIndex = ndIndex
    for i in 0..<ndIndex.count {
        if ndIndex[i] < 0 {
            ndIndex[i] += shape[i]
        }
        precondition(0 <= ndIndex[i] && ndIndex[i] < shape[i], "Index is not in valid range.")
    }
    return ndIndex
}

/// Return normalized axis.
/// - Check axis is in valid range
/// - Process minus number
func normalizeAxis(axis: Int, ndim: Int) -> Int {
    assert(ndim > 0)
    var axis = axis
    if axis < 0 {
        axis += ndim
    }
    precondition(axis >= 0 && axis < ndim, "Axis is not in valid range.")
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
