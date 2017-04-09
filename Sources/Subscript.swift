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
    
    let startNDIndex = expandedIndex.map { $0 ?? 0 }
    let startIndex = normalizeIndex(shape: ndarray.shape, ndIndex: startNDIndex)
    
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
    
    let count = Int32(minorShape.reduce(1, *))
    let srcStride = Int32(newValue.strides.last ?? 0)
    let dstStride = Int32(dstStrides.last ?? 0)
    
    for majorIndex in NDIndexSequence(shape: majorShape) {
        let ndIndex = majorIndex + minorZeros
        let src = UnsafePointer(newValue.data)
            .advanced(by: indexOffset(strides: newValue.strides, ndIndex: ndIndex) + newValue.baseOffset)
        let dst = UnsafeMutablePointer(mutating: newData)
            .advanced(by: dstOffset + indexOffset(strides: dstStrides, ndIndex: ndIndex))
        cblas_scopy(count, src, srcStride, dst, dstStride)
    }
    
    ndarray = NDArray(shape: ndarray.shape, strides: newStrides, baseOffset: 0, data: newData)
}
