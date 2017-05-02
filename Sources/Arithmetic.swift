import Foundation
import Accelerate

// MARK: - negate
public prefix func -(arg: NDArray) -> NDArray {
    return neg(arg)
}

func neg(_ arg: NDArray) -> NDArray {
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = arg.startPointer
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
        defer { dst.deallocate(capacity: arg.data.count) }
        
        vDSP_vneg(src, 1, dst, 1, vDSP_Length(arg.data.count))
        
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: arg.data.count)))
    } else {
        let ndim = arg.shape.count
        let volume = arg.shape.prod()
        
        let axis = getLeastStrideAxis(arg.strides)
        let srcStride = arg.strides[axis]
        let dims = getStridedDims(shape: arg.shape, strides: arg.strides, from: axis)
        
        let outerShape = [Int](arg.shape[0..<axis-dims+1] + arg.shape[axis+1..<ndim])
        let outerStrides = [Int](arg.strides[0..<axis-dims+1] + arg.strides[axis+1..<ndim])
        let blockSize = arg.shape[axis-dims+1...axis].prod()
        
        let dstStrides = contiguousStrides(shape: arg.shape)
        let dstOuterStrides = [Int](dstStrides[0..<axis-dims+1] + dstStrides[axis+1..<ndim])
        
        let dstStride = dstStrides[axis]
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let srcOffsets = getOffsets(shape: outerShape, strides: outerStrides)
        let dstOffsets = getOffsets(shape: outerShape, strides: dstOuterStrides)
        let _blockSize = vDSP_Length(blockSize)
        
        let src = arg.startPointer
        for (os, od) in zip(srcOffsets, dstOffsets) {
            let src = src + os
            let dst = dst + od
            
            vDSP_vneg(src, srcStride, dst, dstStride, _blockSize)
        }

        return NDArray(shape: arg.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}

// MARK: - NDArray and Scalar

public func +(lhs: NDArray, rhs: Float) -> NDArray {
    return apply(lhs, rhs, vDSP_vsadd)
}

public func -(lhs: NDArray, rhs: Float) -> NDArray {
    return apply(lhs, -rhs, vDSP_vsadd)
}

public func *(lhs: NDArray, rhs: Float) -> NDArray {
    return apply(lhs, rhs, vDSP_vsmul)
}

public func /(lhs: NDArray, rhs: Float) -> NDArray {
    return apply(lhs, rhs, vDSP_vsdiv)
}

public func +(lhs: Float, rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vDSP_vsadd)
}

public func -(lhs: Float, rhs: NDArray) -> NDArray {
    return apply(-rhs, lhs, vDSP_vsadd)
}

public func *(lhs: Float, rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vDSP_vsmul)
}

public func /(lhs: Float, rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_svdiv)
}

// MARK: Util

private typealias vDSP_vs_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafePointer<Float>,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

private typealias vDSP_sv_func = (UnsafePointer<Float>,
    UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

private func apply(_ lhs: NDArray,
                   _ rhs: Float,
                   _ vDSPfunc: vDSP_vs_func) -> NDArray {
    
    let strDims = stridedDims(shape: lhs.shape, strides: lhs.strides)
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let minorShape = lhs.shape.suffix(strDims)
    
    let blockSize = minorShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: lhs.volume)
    defer { dst.deallocate(capacity: lhs.volume) }
    
    let majorStrides = [Int](lhs.strides.dropLast(strDims))
    let offsets = getOffsets(shape: majorShape, strides: majorStrides)
    
    var rhs = rhs
    let stride = vDSP_Stride(lhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let src = lhs.startPointer
    var dstPtr = dst
    for offset in offsets {
        let src = src + offset
        vDSPfunc(src, stride,
                 &rhs,
                 dstPtr, 1, _blockSize)
        
        dstPtr += blockSize
    }
    
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: lhs.volume)))
}

private func apply(_ lhs: Float,
                   _ rhs: NDArray,
                   _ vDSPfunc: vDSP_sv_func) -> NDArray {
    
    let strDims = stridedDims(shape: rhs.shape, strides: rhs.strides)
    
    let majorShape = [Int](rhs.shape.dropLast(strDims))
    let minorShape = rhs.shape.suffix(strDims)
    
    let blockSize = minorShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: rhs.volume)
    defer { dst.deallocate(capacity: rhs.volume) }
    
    let majorStrides = [Int](rhs.strides.dropLast(strDims))
    let offsets = getOffsets(shape: majorShape, strides: majorStrides)
    
    var lhs = lhs
    let stride = vDSP_Stride(rhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let src = rhs.startPointer
    var dstPtr = dst
    for offset in offsets {
        let src = src + offset
        vDSPfunc(&lhs,
                 src, stride,
                 dstPtr, 1, _blockSize)
        
        dstPtr += blockSize
    }
    
    return NDArray(shape: rhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: rhs.volume)))
}

// MARK: - NDArray and NDArray
public func +(lhs: NDArray, rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_vadd)
}

public func -(lhs: NDArray, rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vDSP_vsub)
}

public func *(lhs: NDArray, rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_vmul)
}

public func /(lhs: NDArray, rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vDSP_vdiv)
}

// MARK: Util
private typealias vDSP_v_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

private func apply(_ lhs: NDArray,
           _ rhs: NDArray,
           _ vDSPfunc: vDSP_v_func) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let strDims = min(stridedDims(shape: lhs.shape, strides: lhs.strides),
                      stridedDims(shape: rhs.shape, strides: rhs.strides))
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let minorShape = lhs.shape.suffix(strDims)
    
    let blockSize = minorShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: lhs.volume)
    defer { dst.deallocate(capacity: lhs.volume) }
    
    let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
    let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
    let lOffsets = getOffsets(shape: majorShape, strides: lMajorStrides)
    let rOffsets = getOffsets(shape: majorShape, strides: rMajorStrides)
    let lStride = vDSP_Stride(lhs.strides.last ?? 0)
    let rStride = vDSP_Stride(rhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let lSrc = lhs.startPointer
    let rSrc = rhs.startPointer
    var dstPtr = dst
    for (lOffset, rOffset) in zip(lOffsets, rOffsets) {
        let lSrc = lSrc + lOffset
        let rSrc = rSrc + rOffset
        vDSPfunc(lSrc, lStride,
                 rSrc, rStride,
                 dstPtr, 1, _blockSize)
        
        dstPtr += blockSize
    }
    
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: lhs.volume)))
}
