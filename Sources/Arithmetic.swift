import Foundation
import Accelerate

// MARK: - negate
public prefix func -(arg: NDArray) -> NDArray {
    return neg(arg)
}

func neg(_ arg: NDArray) -> NDArray {
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = UnsafePointer(arg.data).advanced(by: arg.baseOffset)
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
        defer { dst.deallocate(capacity: arg.data.count) }
        vDSP_vneg(src, 1, dst, 1, vDSP_Length(arg.data.count))
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: arg.data.count)))
    } else {
        // Separate scattered major shape and strided minor shape
        let volume = arg.volume
        let minorDims = stridedDims(shape: arg.shape, strides: arg.strides)
        let majorShape = [Int](arg.shape.dropLast(minorDims))
        let majorStrides = [Int](arg.strides.dropLast(minorDims))
        
        let stride = arg.strides.last!
        let blockSize = arg.shape.suffix(minorDims).prod()
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let offsets = getOffsets(shape: majorShape, strides: majorStrides)
        let _blockSize = vDSP_Length(blockSize)
        
        let src = UnsafePointer(arg.data) + arg.baseOffset
        var dstPtr = dst
        for offset in offsets {
            let srcPtr = src + offset
            
            vDSP_vneg(srcPtr, stride, dstPtr, 1, _blockSize)
            dstPtr += blockSize
        }
        
        return NDArray(shape: arg.shape, elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
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
    
    var dst = UnsafeMutablePointer<Float>.allocate(capacity: lhs.volume)
    defer { dst.deallocate(capacity: lhs.volume) }
    
    let majorStrides = [Int](lhs.strides.dropLast(strDims))
    let offsets = getOffsets(shape: majorShape, strides: majorStrides)
    
    var rhs = rhs
    let stride = vDSP_Stride(lhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let src = UnsafePointer(lhs.data) + lhs.baseOffset
    var dstPtr = dst
    for offset in offsets {
        let srcPtr = src + offset
        vDSPfunc(srcPtr, stride,
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
    
    var dst = UnsafeMutablePointer<Float>.allocate(capacity: rhs.volume)
    defer { dst.deallocate(capacity: rhs.volume) }
    
    let majorStrides = [Int](rhs.strides.dropLast(strDims))
    let offsets = getOffsets(shape: majorShape, strides: majorStrides)
    
    var lhs = lhs
    let stride = vDSP_Stride(rhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let src = UnsafePointer(rhs.data) + rhs.baseOffset
    var dstPtr = dst
    for offset in offsets {
        let srcPtr = src + offset
        vDSPfunc(&lhs,
                 srcPtr, stride,
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
    
    var dst = UnsafeMutablePointer<Float>.allocate(capacity: lhs.volume)
    defer { dst.deallocate(capacity: lhs.volume) }
    
    let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
    let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
    let lOffsets = getOffsets(shape: majorShape, strides: lMajorStrides)
    let rOffsets = getOffsets(shape: majorShape, strides: rMajorStrides)
    let lStride = vDSP_Stride(lhs.strides.last ?? 0)
    let rStride = vDSP_Stride(rhs.strides.last ?? 0)
    let _blockSize = vDSP_Length(blockSize)
    
    let lSrc = UnsafePointer(lhs.data) + lhs.baseOffset
    let rSrc = UnsafePointer(rhs.data) + rhs.baseOffset
    var dstPtr = dst
    for (lOffset, rOffset) in zip(lOffsets, rOffsets) {
        let lSrcPtr = lSrc + lOffset
        let rSrcPtr = rSrc + rOffset
        vDSPfunc(lSrcPtr, lStride,
                 rSrcPtr, rStride,
                 dstPtr, 1, _blockSize)
        
        dstPtr += blockSize
    }
    
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: lhs.volume)))
}
