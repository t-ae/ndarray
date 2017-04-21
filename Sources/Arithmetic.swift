import Foundation
import Accelerate

// MARK: - add
public func +(lhs: NDArray, rhs: NDArray) -> NDArray {
    return add(lhs, rhs)
}

func add(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let strDims = min(stridedDims(shape: lhs.shape, strides: lhs.strides),
                      stridedDims(shape: rhs.shape, strides: rhs.strides))
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let minorShape = lhs.shape.suffix(strDims)
    
    let count = minorShape.reduce(1, *)
    
    var dst = UnsafeMutablePointer<Float>.allocate(capacity: lhs.volume)
    defer { dst.deallocate(capacity: lhs.volume) }
    
    let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
    let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
    var lOffsets = getOffsets(shape: majorShape, strides: lMajorStrides)
    var rOffsets = getOffsets(shape: majorShape, strides: rMajorStrides)
    let lStride = Int32(lhs.strides.last ?? 0)
    let rStride = Int32(rhs.strides.last ?? 0)
    
    let numProc = ProcessInfo.processInfo.activeProcessorCount
    let blockSize = Int(ceil(Float(lOffsets.count) / Float(numProc)))
    
    DispatchQueue.concurrentPerform(iterations: numProc) { i in
        var dstPtr = dst.advanced(by: i*blockSize*count)
        let end = i*blockSize + min(blockSize, lOffsets.count - i*blockSize)
        
        guard i*blockSize < end else { // can be empty
            return
        }
        
        for oi in i*blockSize..<end {
            let lOffset = lOffsets[oi] + lhs.baseOffset
            let lSrc = UnsafePointer(lhs.data).advanced(by: lOffset)
            let rOffset = rOffsets[oi] + rhs.baseOffset
            let rSrc = UnsafePointer(rhs.data).advanced(by: rOffset)
            vDSP_vadd(lSrc, vDSP_Stride(lStride),
                      rSrc, vDSP_Stride(rStride),
                      dstPtr, 1, vDSP_Length(count))
            dstPtr += count
        }
    }
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: lhs.volume)))
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
