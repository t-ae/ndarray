
import Accelerate

// MARK: - Unary

typealias vDSP_unary_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ arg: NDArray, _ vDSPfunc: vDSP_unary_func) -> NDArray {
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = arg.startPointer
        var dst = NDArrayData(size: arg.data.count)
        
        dst.withUnsafeMutablePointer { p in
            vDSPfunc(src, 1, p, 1, vDSP_Length(arg.data.count))
        }
        
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: dst)
    } else {
        let ndim = arg.ndim
        let volume = arg.volume
        
        let axis = getLeastStrideAxis(arg.strides)
        let srcStride = arg.strides[axis]
        let dims = getStridedDims(shape: arg.shape, strides: arg.strides, from: axis)
        
        let outerShape = [Int](arg.shape[0..<axis-dims+1] + arg.shape[axis+1..<ndim])
        let outerStrides = [Int](arg.strides[0..<axis-dims+1] + arg.strides[axis+1..<ndim])
        let blockSize = arg.shape[axis-dims+1...axis].prod()
        
        let dstStrides = getContiguousStrides(shape: arg.shape)
        let dstOuterStrides = [Int](dstStrides[0..<axis-dims+1] + dstStrides[axis+1..<ndim])
        
        let dstStride = dstStrides[axis]
        
        var dst = NDArrayData(size: volume)
        
        let offsets = BinaryOffsetSequence(shape: outerShape, lStrides: outerStrides, rStrides: dstOuterStrides)
        let _blockSize = vDSP_Length(blockSize)
        
        let src = arg.startPointer
        dst.withUnsafeMutablePointer { dstHead in
            for (os, od) in offsets {
                let src = src + os
                let dst = dstHead + od
                
                vDSPfunc(src, srcStride, dst, dstStride, _blockSize)
            }
        }
        
        return NDArray(shape: arg.shape, elements: dst)
    }
}

// MARK: - Binary
// MARK: Vector-Scalar operation
typealias vDSP_vs_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafePointer<Float>,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ lhs: NDArray,
           _ rhs: Float,
           _ vDSPfunc: @escaping vDSP_vs_func) -> NDArray {
    var rhs = rhs
    let f: vDSP_unary_func = { sp, ss, dp, ds, len in
        vDSPfunc(sp, ss, &rhs, dp, ds, len)
    }
    return apply(lhs, f)
}

// MARK: Scalar-Vector operation
typealias vDSP_sv_func = (UnsafePointer<Float>,
    UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ lhs: Float,
           _ rhs: NDArray,
           _ vDSPfunc: @escaping vDSP_sv_func) -> NDArray {
    var lhs = lhs
    let f: vDSP_unary_func = { sp, ss, dp, ds, len in
        vDSPfunc(&lhs, sp, ss, dp, ds, len)
    }
    return apply(rhs, f)
}

// MARK: Vector-Vector operation
typealias vDSP_vv_func = (UnsafePointer<Float>, vDSP_Stride,
    UnsafePointer<Float>, vDSP_Stride,
    UnsafeMutablePointer<Float>, vDSP_Stride, vDSP_Length) -> Void

func apply(_ lhs: NDArray, _ rhs: NDArray, _ vDSPfunc: vDSP_vv_func) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    var dst = NDArrayData(size: volume)
    
    let strDims = min(getStridedDims(shape: lhs.shape, strides: lhs.strides),
                      getStridedDims(shape: rhs.shape, strides: rhs.strides))
    
    let majorShape = [Int](lhs.shape.dropLast(strDims))
    let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
    let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
    
    let lStride = vDSP_Stride(lhs.strides.last ?? 0)
    let rStride = vDSP_Stride(rhs.strides.last ?? 0)
    let blockSize = lhs.shape.suffix(strDims).prod()
    let _blockSize = vDSP_Length(blockSize)
    
    let offsets = BinaryOffsetSequence(shape: majorShape, lStrides: lMajorStrides, rStrides: rMajorStrides)
    
    let lSrc = lhs.startPointer
    let rSrc = rhs.startPointer
    dst.withUnsafeMutablePointer { dst in
        var dst = dst
        for (lo, ro) in offsets {
            vDSPfunc(lSrc + lo, lStride,
                     rSrc + ro, rStride,
                     dst, 1, _blockSize)
            dst += blockSize
        }
    }
    
    return NDArray(shape: lhs.shape, elements: dst)
}

// MARK: - Reduce

typealias vDSP_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, vDSP_Length) -> Void

// Reduce all elements.
func reduce(_ arg: NDArray, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    let elements = gatherElements(arg)
    var result: Float = 0
    vDSPfunc(elements.pointer, 1, &result, vDSP_Length(elements.count))
    return NDArray(scalar: result)
}

/// Reduce along a given axis.
func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_reduce_func) -> NDArray {
    
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.prod()
    
    var dst = NDArrayData(size: volume)
    
    let offsets = OffsetSequence(shape: newShape, strides: arg.strides.removing(at: axis))
    let count = vDSP_Length(arg.shape[axis])
    let stride = arg.strides[axis]
    
    let src = arg.startPointer
    dst.withUnsafeMutablePointer { dst in
        var dst = dst
        for offset in offsets {
            let src = src + offset
            vDSPfunc(src, stride, dst, count)
            dst += 1
        }
    }
    
    return NDArray(shape: newShape, elements: dst)
}

typealias vDSP_index_reduce_func = (UnsafePointer<Float>, vDSP_Stride, UnsafeMutablePointer<Float>, UnsafeMutablePointer<vDSP_Length>, vDSP_Length) -> Void

// Reduce along a given axis (for argmin, argmax).
func reduce(_ arg: NDArray, along axis: Int, _ vDSPfunc: vDSP_index_reduce_func) -> NDArray {
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let newShape = arg.shape.removing(at: axis)
    let volume = newShape.prod()
    
    let dst = [vDSP_Length](repeating: 0, count: volume)
    var e: Float = 0
    
    let offsets = OffsetSequence(shape: newShape, strides: arg.strides.removing(at: axis))
    let count = vDSP_Length(arg.shape[axis])
    let stride = arg.strides[axis]
    
    var dstPtr = UnsafeMutablePointer(mutating: dst)
    let src = arg.startPointer
    for offset in offsets {
        let src = src + offset
        vDSPfunc(src, stride, &e, dstPtr, count)
        dstPtr += 1
    }
    
    var elements = NDArrayData(size: volume)
    elements.withUnsafeMutablePointer { p in
        var p = p
        let stride = UInt(stride)
        for i in dst {
            // all indices are multiplied with stride.
            p.pointee = Float(i/stride)
            p += 1
        }
    }
    
    return NDArray(shape: newShape, elements: elements)
}
