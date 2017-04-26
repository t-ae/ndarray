import Foundation
import Accelerate

extension NDArray {
    
    public subscript(index: Int?...) -> NDArray {
        get {
            return get(array: self, indexWithHole: index)
        }
        set {
            set(array: &self, indexWithHole: index, newValue: newValue)
        }
    }
    
}

func get(array: NDArray, indexWithHole: [Int?]) -> NDArray {
    
    precondition(indexWithHole.count <= array.ndim)
    
    // fill rest dimensions
    let expandedIndex = indexWithHole + [Int?](repeating: nil, count: array.ndim - indexWithHole.count)
    
    let newShape = zip(array.shape, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    let newStrides = zip(array.strides, expandedIndex).filter { $0.1 == nil }.map { $0.0 }
    
    let startIndex = normalizeIndex(shape: array.shape, ndIndex: expandedIndex.map { $0 ?? 0 })
    
    let newOffset = indexOffset(strides: array.strides, ndIndex: startIndex) + array.baseOffset
    return NDArray(shape: newShape, strides: newStrides, baseOffset: newOffset, data: array.data)
}

func set(array: inout NDArray, indexWithHole: [Int?], newValue: NDArray) {
    
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
