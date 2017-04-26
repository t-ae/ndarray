import Foundation
import Accelerate

extension NDArray {
    
    public subscript(index: Int?...) -> NDArray {
        get {
            return get(ndarray: self, indexWithHole: index)
        }
        set {
            set(ndarray: &self, indexWithHole: index, newValue: newValue)
        }
    }
    
}

func get(ndarray: NDArray, indexWithHole: [Int?]) -> NDArray {
    
    precondition(indexWithHole.count <= ndarray.ndim)
    
    // fill rest dimensions
    let expandedIndex = indexWithHole + [Int?](repeating: nil, count: ndarray.ndim - indexWithHole.count)
    
    let newShape = zip(ndarray.shape, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    let newStrides = zip(ndarray.strides, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    let startIndex = normalizeIndex(shape: ndarray.shape, ndIndex: expandedIndex.map { $0 ?? 0 })
    
    let newOffset = indexOffset(strides: ndarray.strides, ndIndex: startIndex) + ndarray.baseOffset
    return NDArray(shape: newShape, strides: newStrides, baseOffset: newOffset, data: ndarray.data)
}

func set(ndarray: inout NDArray, indexWithHole: [Int?], newValue: NDArray) {
    
    precondition(indexWithHole.count <= ndarray.ndim)
    
    // fill rest dimensions
    let expandedIndex = indexWithHole + [Int?](repeating: nil, count: ndarray.ndim - indexWithHole.count)

    let dstShape = zip(ndarray.shape, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    // broadcast
    let newValue = broadcast(newValue, to: dstShape)
    
    let startNDIndex = expandedIndex.map { $0 ?? 0 }
    let startIndex = normalizeIndex(shape: ndarray.shape, ndIndex: startNDIndex)
    
    let newData = gatherElements(ndarray, forceUniqueReference: true)
    let newStrides = continuousStrides(shape: ndarray.shape)
    let dstOffset = indexOffset(strides: newStrides, ndIndex: startIndex)
    let dstStrides = zip(newStrides, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    let strDims = min(stridedDims(shape: dstShape, strides: dstStrides),
                      stridedDims(shape: newValue.shape, strides: newValue.strides))
    
    let majorShape = [Int](dstShape.dropLast(strDims))
    let minorShape = dstShape.suffix(strDims)
    let minorZeros = [Int](repeating: 0, count: minorShape.count)
    
    let count = Int32(minorShape.prod())
    let srcStride = Int32(newValue.strides.last ?? 0)
    let dstStride = Int32(dstStrides.last ?? 0)
    
    let majorIndices = getIndices(shape: majorShape)
    
    let src = newValue.startPointer
    let dst = UnsafeMutablePointer(mutating: newData) + dstOffset
    for majorIndex in majorIndices {
        let ndIndex = majorIndex + minorZeros
        let src = src + indexOffset(strides: newValue.strides, ndIndex: ndIndex)
        let dst = dst + indexOffset(strides: dstStrides, ndIndex: ndIndex)
        cblas_scopy(count, src, srcStride, dst, dstStride)
    }
    
    ndarray.strides = newStrides
    ndarray.baseOffset = 0
    ndarray.data = newData
}
