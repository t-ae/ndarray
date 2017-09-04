
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
    
    guard !shape.isEmpty else {
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

func denseDataCount(shape: [Int], strides: [Int]) -> Int {
    assert(isDense(shape: shape, strides: strides))
    var ret = 1
    for (sh, st) in zip(shape, strides) {
        if st != 0 {
            ret *= sh
        }
    }
    return ret
}

/// Get contiguous strides.
func getContiguousStrides(shape: [Int]) -> [Int] {
    assert(shape.all { $0 >= 0 })
    guard !shape.isEmpty else {
        return []
    }
    var stride = 1
    var strides = [Int](repeating: 1, count: shape.count)
    for (i, s) in shape.dropFirst().reversed().enumerated() {
        stride *= s
        strides[shape.count-i-2] = stride
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
func gatherElements(_ arg: NDArray) -> [Float] {
    
    let arg = arg.squeezed()
    let volume = arg.volume
    
    if isContiguous(shape: arg.shape, strides: arg.strides) {
        if volume == arg.data.count {
            return arg.data
        } else {
            let start = arg.baseOffset
            let end = start + volume
            return [Float](arg.data[start..<end])
        }
    } else {

        let axis = getLeastStrideAxis(arg.strides)
        let dims = getStridedDims(shape: arg.shape, strides: arg.strides, from: axis)
        
        let dstStrides = getContiguousStrides(shape: arg.shape)
        
        let offsets = createBinaryOffsetSequence(shape: arg.shape,
                                                 lStrides: arg.strides, rStrides: dstStrides,
                                                 axis: axis, dims: dims)
        
        var dst = [Float](repeating: 0, count: volume)
        
        arg.withUnsafePointer { p in
            dst.withUnsafeMutableBufferPointer {
                copyElements(src: p,
                             srcStride: arg.strides[axis],
                             dst: $0.baseAddress!,
                             dstStride: dstStrides[axis],
                             blockSize: arg.shape[axis-dims+1...axis].prod(),
                             offsets: offsets)
            }
        }
        
        return dst
    }
}

func copyElements(src: UnsafePointer<Float>,
                  srcStride: Int,
                  dst: UnsafeMutablePointer<Float>,
                  dstStride: Int,
                  blockSize: Int,
                  offsets: BinaryOffsetSequence) {
    var src = src
    if srcStride < 0 {
        src += (blockSize-1) * srcStride
    }
    var dst = dst
    if dstStride < 0 {
        dst += (blockSize-1) * dstStride
    }
    let _blockSize = Int32(blockSize)
    let _srcStride = Int32(srcStride)
    let _dstStride = Int32(dstStride)
    for (os, od) in offsets {
        cblas_scopy(_blockSize,
                    src + os,
                    _srcStride,
                    dst + od,
                    _dstStride)
    }
}

func createBinaryOffsetSequence(shape: [Int],
                                lStrides: [Int],
                                rStrides: [Int],
                                axis: Int,
                                dims: Int) -> BinaryOffsetSequence {
    
    assert(lStrides.count == rStrides.count)
    assert(0 <= axis && axis < lStrides.count)
    assert(0 <= axis - dims + 1)
    
    let ndim = lStrides.count
    
    if axis == ndim-1 {
        let outerShape = shape[0..<axis-dims+1]
        let outerLStrides = lStrides[0..<axis-dims+1]
        let outerRStrides = rStrides[0..<axis-dims+1]
        
        return BinaryOffsetSequence(shape: outerShape, lStrides: outerLStrides, rStrides: outerRStrides)
    } else if axis - dims + 1 == 0 {
        let outerShape = shape[axis+1..<ndim]
        let outerLStrides = lStrides[axis+1..<ndim]
        let outerRStrides = rStrides[axis+1..<ndim]
        
        return BinaryOffsetSequence(shape: outerShape, lStrides: outerLStrides, rStrides: outerRStrides)
    } else {
        let outerShape = [Int](shape[0..<axis-dims+1] + shape[axis+1..<ndim])
        let outerLStrides = [Int](lStrides[0..<axis-dims+1] + lStrides[axis+1..<ndim])
        let outerRStrides = [Int](rStrides[0..<axis-dims+1] + rStrides[axis+1..<ndim])
        
        return BinaryOffsetSequence(shape: outerShape, lStrides: outerLStrides, rStrides: outerRStrides)
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
    @inline(__always)
    func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        return try data.withUnsafeBufferPointer {
            try body($0.baseAddress! + baseOffset)
        }
    }
}

// MARK: - Standard library extensions
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
    
    func appending(_ newElement: Element) -> Array {
        var ret = self
        ret.append(newElement)
        return ret
    }
    
    func inserting(_ newElement: Element, at: Int) -> Array {
        var ret = self
        ret.insert(newElement, at: at)
        return ret
    }
}

extension ArraySlice {
    func all(cond: (Element)->Bool) -> Bool {
        for e in self {
            if !cond(e) {
                return false
            }
        }
        return true
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
