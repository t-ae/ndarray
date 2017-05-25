import Foundation
import Accelerate

extension NDArray {
    
    public subscript(indices: NDArrayIndexElementProtocol?...) -> NDArray {
        get {
            return getSubarray(array: self, indices: indices)
        }
        set {
            setSubarray(array: &self, indices: indices, newValue: newValue)
        }
    }

    /// Substitute for scalar setting
    public mutating func set(_ value: Float, for index: [Int?]) {
        setSubarray(array: &self, indices: index, newValue: NDArray(scalar: value))
    }
}

//MARK: - Implementation
func getSubarray(array: NDArray, indices: [NDArrayIndexElementProtocol?]) -> NDArray {
    precondition(indices.count <= array.ndim, "Too many indices for NDArray.")
    
    let indices = indices.map { $0.map(toNDArrayIndexElement) }
    
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

func setSubarray(array: inout NDArray, indices: [NDArrayIndexElementProtocol?], newValue: NDArray) {
    
    precondition(indices.count <= array.ndim, "Too many indices for NDArray.")
    
    let indices = indices.map { $0.map(toNDArrayIndexElement) }
    
    // Make array contiguous
    array.data = gatherElements(array)
    array.strides = getContiguousStrides(shape: array.shape)
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
    
    let strDims = min(getStridedDims(shape: dstShape, strides: dstStrides),
                      getStridedDims(shape: newValue.shape, strides: newValue.strides))
    
    let majorShape = [Int](dstShape.dropLast(strDims))
    
    let srcMajorStrides = [Int](newValue.strides.dropLast(strDims))
    let dstMajorStrides = [Int](dstStrides.dropLast(strDims))
    
    let offsets = BinaryOffsetSequence(shape: majorShape, lStrides: srcMajorStrides, rStrides: dstMajorStrides)
    
    let blockSize = dstShape.suffix(strDims).prod()
    let srcStride = Int32(newValue.strides.last ?? 1)
    let dstStride = Int32(dstStrides.last ?? 1)
    
    let _blockSize = Int32(blockSize)
    let src: UnsafePointer<Float>
    if srcStride < 0 {
        src = newValue.startPointer + (blockSize-1)*Int(srcStride)
    } else {
        src = newValue.startPointer
    }
    // Force CoW
    array.data.withUnsafeMutablePointer { p in
        let dst: UnsafeMutablePointer<Float>
        if dstStride < 0 {
            dst = p + (array.baseOffset + dstOffset + (blockSize-1)*Int(dstStride))
        } else {
            dst = p + (array.baseOffset + dstOffset)
        }
        for (os, od) in offsets {
            let src = src + os
            let dst = dst + od
            cblas_scopy(_blockSize, src, srcStride, dst, dstStride)
        }
    }
}
