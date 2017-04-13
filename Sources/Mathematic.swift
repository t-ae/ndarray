import Foundation
import Accelerate

// MARK: - add
public func +(lhs: NDArray, rhs: NDArray) -> NDArray {
    return add(lhs, rhs)
}

func add(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let y = gatherElements(rhs, forceUniqueReference: true)
    
    let strDims = stridedDims(shape: lhs.shape, strides: lhs.strides)
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let majorStrides = [Int](lhs.strides.dropLast(strDims))
    let minorShape = lhs.shape.suffix(strDims)
    
    let count = minorShape.reduce(1, *)
    var dst = UnsafeMutablePointer(mutating: y)
    
    let xStride = Int32(lhs.strides.last ?? 0)
    
    for majorIndex in NDIndexSequence(shape: majorShape) {
        let offset = indexOffset(strides: majorStrides, ndIndex: majorIndex) + lhs.baseOffset
        let src = UnsafePointer(lhs.data) + offset
        cblas_saxpy(Int32(count), 1,
                    src, xStride,
                    dst, 1)
        dst += count
    }
    
    return NDArray(shape: lhs.shape, elements: y)
}

// MARK: - negate
public prefix func -(arg: NDArray) -> NDArray {
    return neg(arg)
}

func neg(_ arg: NDArray) -> NDArray {
    let volume = arg.volume
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = UnsafePointer(arg.data).advanced(by: arg.baseOffset)
        var dst = [Float](repeating: 0, count: volume)
        cblas_saxpy(Int32(volume), -1,
                    src, 1,
                    &dst, 1)
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: dst)
    } else {
        let elements = gatherElements(arg)
        
        var dst = [Float](repeating: 0, count: volume)
        cblas_saxpy(Int32(volume), -1,
                    elements, 1,
                    &dst, 1)
        return NDArray(shape: arg.shape, elements: dst)
    }
}

// MARK: - sqrt
public func sqrt(_ arg: NDArray) -> NDArray {
    return _sqrt(arg)
}

func _sqrt(_ arg: NDArray) -> NDArray {
    
    let volume = arg.volume
    var count = Int32(volume)
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = UnsafePointer(arg.data).advanced(by: arg.baseOffset)
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        vvsqrtf(dst, src, &count)
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    } else {
        let elements = gatherElements(arg)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        
        vvsqrtf(dst, elements, &count)
        return NDArray(shape: arg.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
