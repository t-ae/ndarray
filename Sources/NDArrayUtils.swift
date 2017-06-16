
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
func gatherElements(_ arg: NDArray) -> NDArrayData<Float> {
    
    let arg = arg.squeezed()
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
        let dims = getStridedDims(shape: arg.shape, strides: arg.strides, from: axis)
    
        let outerShape = [Int](arg.shape[0..<axis-dims+1] + arg.shape[axis+1..<ndim])
        let outerStrides = [Int](arg.strides[0..<axis-dims+1] + arg.strides[axis+1..<ndim])
        
        let dstStrides = getContiguousStrides(shape: arg.shape)
        let dstOuterStrides = [Int](dstStrides[0..<axis-dims+1] + dstStrides[axis+1..<ndim])
        
        var dst = NDArrayData<Float>(size: volume)
        
        arg.withUnsafePointer { p in
            dst.withUnsafeMutablePointer {
                copyElements(src: p,
                             srcStride: arg.strides[axis],
                             dst: $0,
                             dstStride: dstStrides[axis],
                             blockSize: arg.shape[axis-dims+1...axis].prod(),
                             offsets: BinaryOffsetSequence(shape: outerShape,
                                                           lStrides: outerStrides,
                                                           rStrides: dstOuterStrides))
            }
        }
        
        return dst
    }
}

@inline(__always)
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
    func withUnsafePointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R{
        return try data.withUnsafePointer {
            try body($0 + baseOffset)
            
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

// MARK: - Pointer combination
// MARK: NDArray
@inline(__always)
func withUnsafePointers<R>(_ array0: NDArray,
                           _ array1: NDArray,
                           _ body: (UnsafePointer<Float>, UnsafePointer<Float>) throws -> R) rethrows -> R {
    return try array0.withUnsafePointer { p0 in
        try array1.withUnsafePointer { p1 in
            try body(p0, p1)
        }
    }
}

// MARK: NDArrayData
func withUnsafePointers<T, R>(_ list: [NDArrayData<T>], _ body: @escaping ([UnsafePointer<T>]) throws -> R) rethrows -> R {
    
    func process(_ list: [NDArrayData<T>], _ ptrs: [UnsafePointer<T>]) throws -> R {
        if list.isEmpty {
            return try body(ptrs)
        } else {
            return try list.first!.withUnsafePointer { p in
                try process(Array(list.dropFirst()), ptrs + [p])
            }
        }
    }
    
    return try process(list, [])
}

@inline(__always)
func withUnsafePointers<T0, T1, R>(_ data0: NDArrayData<T0>,
                                   _ data1: NDArrayData<T1>,
                                   _ body: (UnsafePointer<T0>, UnsafePointer<T1>) throws -> R) rethrows -> R {
    return try data0.withUnsafePointer { p0 in
        try data1.withUnsafePointer { p1 in
            try body(p0, p1)
        }
    }
}

@inline(__always)
func withUnsafeMutablePointers<T0, T1, R>(_ data0: inout NDArrayData<T0>,
                                          _ data1: inout NDArrayData<T1>,
                                          _ body: (UnsafeMutablePointer<T0>, UnsafeMutablePointer<T1>) throws -> R) rethrows -> R {
    return try data0.withUnsafeMutablePointer { p0 in
        try data1.withUnsafeMutablePointer { p1 in
            try body(p0, p1)
        }
    }
}

@inline(__always)
func withUnsafeMutablePointers<T0, T1, T2, R>(_ data0: inout NDArrayData<T0>,
                                              _ data1: inout NDArrayData<T1>,
                                              _ data2: inout NDArrayData<T2>,
                                              _ body: (UnsafeMutablePointer<T0>, UnsafeMutablePointer<T1>, UnsafeMutablePointer<T2>) throws -> R) rethrows -> R {
    return try data0.withUnsafeMutablePointer { p0 in
        try data1.withUnsafeMutablePointer { p1 in
            try data2.withUnsafeMutablePointer { p2 in
                try body(p0, p1, p2)
            }
        }
    }
}

@inline(__always)
func withUnsafeMutablePointers<T0, T1, T2, T3, R>(_ data0: inout NDArrayData<T0>,
                                                  _ data1: inout NDArrayData<T1>,
                                                  _ data2: inout NDArrayData<T2>,
                                                  _ data3: inout NDArrayData<T3>,
                                                  _ body: (UnsafeMutablePointer<T0>, UnsafeMutablePointer<T1>, UnsafeMutablePointer<T2>, UnsafeMutablePointer<T3>) throws -> R) rethrows -> R {
    return try data0.withUnsafeMutablePointer { p0 in
        try data1.withUnsafeMutablePointer { p1 in
            try data2.withUnsafeMutablePointer { p2 in
                try data3.withUnsafeMutablePointer { p3 in
                    try body(p0, p1, p2, p3)
                }
            }
        }
    }
}
