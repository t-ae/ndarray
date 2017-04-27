import Foundation
import Accelerate

extension NDArray {
    
    public subscript(index: Int?...) -> NDArray {
        get {
            return getSubarray(array: self, indexWithHole: index)
        }
        set {
            setSubarray(array: &self, indexWithHole: index, newValue: newValue)
        }
    }
    
    /// Substitute for scalar setting
    public mutating func set(_ value: Float, for index: [Int?]) {
        setSubarray(array: &self, indexWithHole: index, newValue: NDArray(scalar: value))
    }
    
}

func getSubarray(array: NDArray, indexWithHole: [Int?]) -> NDArray {
    
    precondition(indexWithHole.count <= array.ndim)
    
    // fill rest dimensions
    let expandedIndex = indexWithHole + [Int?](repeating: nil, count: array.ndim - indexWithHole.count)
    
    let newShape = zip(array.shape, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    let newStrides = zip(array.strides, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    let startIndex = normalizeIndex(shape: array.shape, ndIndex: expandedIndex.map { $0 ?? 0 })
    
    let newOffset = indexOffset(strides: array.strides, ndIndex: startIndex) + array.baseOffset
    return NDArray(shape: newShape, strides: newStrides, baseOffset: newOffset, data: array.data)
}

func setSubarray(array: inout NDArray, indexWithHole: [Int?], newValue: NDArray) {
    
    precondition(indexWithHole.count <= array.ndim)
    
    // fill rest dimensions
    let expandedIndex = indexWithHole + [Int?](repeating: nil, count: array.ndim - indexWithHole.count)

    let dstShape = zip(array.shape, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    // broadcast
    let newValue = broadcast(newValue, to: dstShape)
    
    let startIndex = normalizeIndex(shape: array.shape,
                                    ndIndex: expandedIndex.map { $0 ?? 0 })
    
    array.data = gatherElements(array, forceUniqueReference: true)
    array.strides = continuousStrides(shape: array.shape)
    array.baseOffset = 0
    
    let dstOffset = indexOffset(strides: array.strides, ndIndex: startIndex)
    let dstStrides = zip(array.strides, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    let strDims = min(stridedDims(shape: dstShape, strides: dstStrides),
                      stridedDims(shape: newValue.shape, strides: newValue.strides))
    
    let majorShape = [Int](dstShape.dropLast(strDims))
    let minorShape = dstShape.suffix(strDims)
    let minorZeros = [Int](repeating: 0, count: minorShape.count)
    
    let blockSize = minorShape.prod()
    let srcStride = Int32(newValue.strides.last ?? 0)
    let dstStride = Int32(dstStrides.last ?? 0)
    
    let majorIndices = getIndices(shape: majorShape)
    
    let _blockSize = Int32(blockSize)
    
    let src: UnsafePointer<Float>
    if srcStride < 0 {
        src = newValue.startPointer - (blockSize-1)
    } else {
        src = newValue.startPointer
    }
    let dst = UnsafeMutablePointer(mutating: array.startPointer) + dstOffset
    for majorIndex in majorIndices {
        let ndIndex = majorIndex + minorZeros
        let src = src + indexOffset(strides: newValue.strides, ndIndex: ndIndex)
        let dst = dst + indexOffset(strides: dstStrides, ndIndex: ndIndex)
        cblas_scopy(_blockSize, src, srcStride, dst, dstStride)
    }
}

// MARK: - For strided subscript
struct NDArrayIndexElement {
    var start: Int?
    var end: Int?
    var stride: Int?
    
    // strided range index
    init(start: Int?, end: Int?, stride: Int?) {
        assert(stride != 0)
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

func getSubarray(array: NDArray, indices: [NDArrayIndexElement?]) -> NDArray {
    precondition(indices.count <= array.ndim)
    
    var x = array
    
    var indices = indices
    indices += [NDArrayIndexElement?](repeating: nil, count: array.ndim - indices.count)
    
    x.shape = []
    x.strides = []
    for i in 0..<array.ndim {
        guard let ie = indices[i] else {
            x.shape.append(array.shape[i])
            x.strides.append(array.strides[i])
            continue
        }
        guard let stride = ie.stride else {
            x.baseOffset += ie.start! * array.strides[i]
            continue
        }
        let start = ie.start ?? 0
        let end = ie.end ?? array.shape[i]
        let size = Int(ceil(abs(Float(end - start) / Float(stride))))
        x.shape.append(size)
        x.strides.append(stride * array.strides[i])
        if stride > 0 {
            x.baseOffset += start * array.strides[i]
        } else {
            x.baseOffset += (end-1) * array.strides[i]
        }
    }
    return x
}
