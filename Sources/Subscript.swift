import Foundation
import Accelerate

extension NDArray {
    
    public subscript() -> NDArray {
        get {
            return getSubarray(array: self, indices: [])
        }
        set {
            setSubarray(array: &self, indices: [], newValue: newValue)
        }
    }
    
    public subscript(indices: NDArrayIndexElement?...) -> NDArray {
        get {
            return getSubarray(array: self, indices: indices)
        }
        set {
            setSubarray(array: &self, indices: indices, newValue: newValue)
        }
    }
    
    public subscript(index: Int?...) -> NDArray {
        get {
            let ies = index.map { $0.map { NDArrayIndexElement(single: $0) } }
            return getSubarray(array: self, indices: ies)
        }
        set {
            let ies = index.map { $0.map { NDArrayIndexElement(single: $0) } }
            setSubarray(array: &self, indices: ies, newValue: newValue)
        }
    }
    
    public subscript(indices: CountableRange<Int>?...) -> NDArray {
        get {
            let ies = indices.map { range in
                range.map { i($0) }
            }
            return getSubarray(array: self, indices: ies)
        }
        set {
            let ies = indices.map { range in
                range.map { i($0) }
            }
            setSubarray(array: &self, indices: ies, newValue: newValue)
        }
    }

    /// Substitute for scalar setting
    public mutating func set(_ value: Float, for index: [Int?]) {
        let ies = index.map { $0.map { NDArrayIndexElement(single: $0) } }
        setSubarray(array: &self, indices: ies, newValue: NDArray(scalar: value))
    }
    
}

public struct NDArrayIndexElement {
    var start: Int?
    var end: Int?
    var stride: Int?
    
    // strided range index
    init(start: Int?, end: Int?, stride: Int?) {
        precondition(stride != 0)
        if let start = start, let end = end {
            precondition((end < 0 && start >= 0) || start <= end, "Invalid range: \(start)..<\(end)")
        }
        self.start = start
        self.end = end
        self.stride = stride ?? 1
    }
    
    // Single index
    init(single: Int) {
        self.start = single
        self.end = nil
        self.stride = nil
    }
}

precedencegroup StridePrecedence {
    associativity: left
    lowerThan: RangeFormationPrecedence
}

infix operator ~~ : StridePrecedence
prefix operator ~~
infix operator ~~- : StridePrecedence
prefix operator ~~-

public func ~~(range: CountableRange<Int>, stride: Int) -> NDArrayIndexElement {
    return i(range, stride)
}

public prefix func ~~(stride: Int) -> NDArrayIndexElement {
    return i(nil, nil, stride)
}

public func ~~-(range: CountableRange<Int>, stride: Int) -> NDArrayIndexElement {
    return range~~(-stride)
}

public prefix func ~~-(stride: Int) -> NDArrayIndexElement {
    return ~~(-stride)
}

public func i(_ range: CountableRange<Int>, _ stride: Int = 1) -> NDArrayIndexElement {
    return i(range.startIndex, range.endIndex, stride)
}
public func i(_ start: Int?, _ end: Int?, _ stride: Int = 1) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: start, end: end, stride: stride)
}

extension NDArrayIndexElement: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init(single: value)
    }
}

func getSubarray(array: NDArray, indices: [NDArrayIndexElement?]) -> NDArray {
    precondition(indices.count <= array.ndim, "Too many indices for NDArray.")
    
    var x = array
    
    x.shape = []
    x.strides = []
    for i in 0..<indices.count {
        guard let ie = indices[i] else {
            x.shape.append(array.shape[i])
            x.strides.append(array.strides[i])
            continue
        }
        var start = ie.start ?? 0
        if start < 0 {
            start += array.shape[i]
        }
        precondition(0 <= start && start < array.shape[i], "Index out of bounds.")
        guard let stride = ie.stride else {
            
            x.baseOffset += start * array.strides[i]
            continue
        }
        var end = ie.end ?? array.shape[i]
        if end < 0 {
            end += array.shape[i]
        }
        precondition(0 <= end && end <= array.shape[i], "Index out of bounds.")
        let size = Int(ceil(abs(Float(end - start) / Float(stride))))
        x.shape.append(size)
        x.strides.append(stride * array.strides[i])
        if stride > 0 {
            x.baseOffset += start * array.strides[i]
        } else {
            x.baseOffset += (end-1) * array.strides[i]
        }
    }
    
    x.shape.append(contentsOf: array.shape.dropFirst(indices.count))
    x.strides.append(contentsOf: array.strides.dropFirst(indices.count))
    
    return x
}

func setSubarray(array: inout NDArray, indices: [NDArrayIndexElement?], newValue: NDArray) {
    
    precondition(indices.count <= array.ndim, "Too many indices for NDArray.")
    
    // Make array continuous
    array.data = gatherElements(array, forceUniqueReference: true)
    array.strides = continuousStrides(shape: array.shape)
    array.baseOffset = 0
    
    // Calculate destinaton
    var dstShape: [Int] = []
    var dstStrides: [Int] = []
    var dstOffset = array.baseOffset
    for i in 0..<indices.count {
        guard let ie = indices[i] else {
            dstShape.append(array.shape[i])
            dstStrides.append(array.strides[i])
            continue
        }
        var start = ie.start ?? 0
        if start < 0 {
            start += array.shape[i]
        }
        precondition(0 <= start && start < array.shape[i], "Index out of bounds.")
        guard let stride = ie.stride else {
            dstOffset += start * array.strides[i]
            continue
        }
        var end = ie.end ?? array.shape[i]
        if end < 0 {
            end += array.shape[i]
        }
        precondition(0 <= end && end <= array.shape[i], "Index out of bounds.")
        let size = Int(ceil(abs(Float(end - start) / Float(stride))))
        dstShape.append(size)
        dstStrides.append(stride * array.strides[i])
        if stride > 0 {
            dstOffset += start * array.strides[i]
        } else {
            dstOffset += (end-1) * array.strides[i]
        }
    }
    dstShape.append(contentsOf: array.shape.dropFirst(indices.count))
    dstStrides.append(contentsOf: array.strides.dropFirst(indices.count))
    
    // Copy
    let newValue = broadcast(newValue, to: dstShape)
    
    let strDims = min(stridedDims(shape: dstShape, strides: dstStrides),
                      stridedDims(shape: newValue.shape, strides: newValue.strides))
    
    let majorShape = [Int](dstShape.dropLast(strDims))
    let minorShape = dstShape.suffix(strDims)
    let minorZeros = [Int](repeating: 0, count: minorShape.count)
    
    let blockSize = minorShape.prod()
    let srcStride = Int32(newValue.strides.last ?? 1)
    let dstStride = Int32(dstStrides.last ?? 1)
    
    let majorIndices = getIndices(shape: majorShape)
    
    let _blockSize = Int32(blockSize)
    let src: UnsafePointer<Float>
    if srcStride < 0 {
        src = newValue.startPointer + (blockSize-1)*Int(srcStride)
    } else {
        src = newValue.startPointer
    }
    let dst: UnsafeMutablePointer<Float>
    if dstStride < 0 {
        dst = UnsafeMutablePointer(mutating: array.startPointer) + dstOffset + (blockSize-1)*Int(dstStride)
    } else {
        dst = UnsafeMutablePointer(mutating: array.startPointer) + dstOffset
    }
    for majorIndex in majorIndices {
        let ndIndex = majorIndex + minorZeros
        let src = src + indexOffset(strides: newValue.strides, ndIndex: ndIndex)
        let dst = dst + indexOffset(strides: dstStrides, ndIndex: ndIndex)
        cblas_scopy(_blockSize, src, srcStride, dst, dstStride)
    }
}
