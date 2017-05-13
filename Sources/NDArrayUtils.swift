
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
        strides = strides.map(-)
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
        strides = strides.map(-)
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
func gatherElements(_ arg: NDArray) -> NDArrayData {
    
    let volume = arg.volume
    let ndim = arg.ndim
    
    if isContiguous(shape: arg.shape, strides: arg.strides) {
        if volume == arg.data.count {
            return arg.data
        } else {
            let start = arg.baseOffset
            let end = start + volume
            return arg.data[start..<end]
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
        
        var dst = NDArrayData(size: volume)
        
        let offsets = BinaryOffsetSequence(shape: outerShape, lStrides: outerStrides, rStrides: dstOuterStrides)
        let _blockSize = Int32(blockSize)
        
        let src: UnsafePointer<Float>
        if srcStride < 0 {
            src = arg.startPointer + (blockSize-1) * Int(srcStride)
        } else {
            src = arg.startPointer
        }
        dst.withUnsafeMutablePointer { dstHead in
            for (os, od) in offsets {
                let src = src + os
                let dst = dstHead + od
                
                cblas_scopy(_blockSize, src, srcStride, dst, dstStride)
            }
        }
        
        return dst
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
        return data.pointer + baseOffset
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
