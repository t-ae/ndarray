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
        let ndim = arg.ndim
        let volume = arg.volume
        
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
