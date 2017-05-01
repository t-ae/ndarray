import Foundation
import Accelerate

extension NDArray {
    
    public subscript(indices: NDArrayIndexElementProtocol?...) -> NDArray {
        get {
            let indices = indices.map { $0.map(toNDArrayIndexElement) }
            return getSubarray(array: self, indices: indices)
        }
        set {
            let indices = indices.map { $0.map(toNDArrayIndexElement) }
            setSubarray(array: &self, indices: indices, newValue: newValue)
        }
    }

    /// Substitute for scalar setting
    public mutating func set(_ value: Float, for index: [Int?]) {
        let ies = index.map { $0.map { NDArrayIndexElement(single: $0) } }
        setSubarray(array: &self, indices: ies, newValue: NDArray(scalar: value))
    }
}

// MARK: - Indexing
public protocol NDArrayIndexElementProtocol { }

public struct NDArrayIndexElement: NDArrayIndexElementProtocol {
    var start: Int?
    var end: Int?
    var stride: Int?
    
    // strided range index
    init(start: Int?, end: Int?, stride: Int = 1) {
        precondition(stride != 0)
        if let start = start, let end = end {
            precondition((end < 0 && start >= 0) || start <= end, "Invalid range: \(start)..<\(end)")
        }
        self.start = start
        self.end = end
        self.stride = stride
    }
    
    // Single index
    init(single: Int) {
        self.start = single
        self.end = nil
        self.stride = nil
    }
}

public struct OneSidedRange: NDArrayIndexElementProtocol {
    var start: Int?
    var end: Int?
    
    init(start: Int?, end: Int?) {
        self.start = start
        self.end = end
    }
}

prefix operator ..<
public prefix func ..<(rhs: Int) -> OneSidedRange {
    return OneSidedRange(start: nil, end: rhs)
}

prefix operator ..<-
public prefix func ..<-(rhs: Int) -> OneSidedRange {
    return OneSidedRange(start: nil, end: -rhs)
}

postfix operator ...
public postfix func ...(lhs: Int) -> OneSidedRange {
    return OneSidedRange(start: lhs, end: nil)
}

extension Int: NDArrayIndexElementProtocol { }
extension CountableRange: NDArrayIndexElementProtocol { }

func toNDArrayIndexElement(_ arg: NDArrayIndexElementProtocol) -> NDArrayIndexElement {
    switch arg {
    case is Int:
        return NDArrayIndexElement(single: arg as! Int)
    case is CountableRange<Int>:
        let arg = arg as! CountableRange<Int>
        return NDArrayIndexElement(start: arg.startIndex, end: arg.endIndex)
    case is OneSidedRange:
        let arg = arg as! OneSidedRange
        return NDArrayIndexElement(start: arg.start, end: arg.end)
    case is NDArrayIndexElement:
        return arg as! NDArrayIndexElement
    default:
        preconditionFailure("\(arg.self) can't convert to NDArrayIndexElement.")
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
    return NDArrayIndexElement(start: range.startIndex, end: range.endIndex, stride: stride)
}

public func ~~(range: OneSidedRange, stride: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: range.start, end: range.end, stride: stride)
}

public prefix func ~~(stride: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: nil, end: nil, stride: stride)
}

public func ~~-(range: CountableRange<Int>, stride: Int) -> NDArrayIndexElement {
    return range~~(-stride)
}

public func ~~-(range: OneSidedRange, stride: Int) -> NDArrayIndexElement {
    return range~~(-stride)
}

public prefix func ~~-(stride: Int) -> NDArrayIndexElement {
    return ~~(-stride)
}


//MARK: - Implementation
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
    
    // Make array contiguous
    array.data = gatherElements(array, forceUniqueReference: true)
    array.strides = contiguousStrides(shape: array.shape)
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
