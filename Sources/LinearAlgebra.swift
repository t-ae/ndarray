
import Foundation
import Accelerate

/// Calculate Frobenius norm.
public func norm(_ arg: NDArray) -> Float {
    return sqrtf(reduce(arg, vDSP_svesq).asScalar())
}

/// Calcurate vector norms along axis.
public func norm(_ arg: NDArray, along axis: Int, keepDims: Bool = false) -> NDArray {
    let ret = sqrt(reduce(arg, along: axis, vDSP_svesq))
    
    return keepDims ? ret.expandDims(axis) : ret
}

/// Calculate determinant.
///
/// If argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two dimensions.
public func determinant(_ arg: NDArray) throws -> NDArray {
    precondition(arg.ndim > 1, "NDArray has shorter ndim(\(arg.ndim)) than 2.")
    let size = arg.shape[arg.ndim-1]
    precondition(arg.shape[arg.ndim-2] == size, "NDArray is not stack of square matrices: shape(\(arg.shape))")
    
    var elements = gatherElements(arg)
    
    let numMatrices = arg.volume / (size*size)
    
    var pivots = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: size)
    defer { pivots.deallocate(capacity: size) }
    
    var N = __CLPK_integer(size)
    let _N = UnsafeMutablePointer(&N)
    
    var error: __CLPK_integer = 0
    
    var out = NDArrayData<Float>(size: numMatrices)
    
    try elements.withUnsafeMutablePointer { ptr in
        try out.withUnsafeMutablePointer { dst in
            var ptr = ptr
            var dst = dst
            for _ in 0..<numMatrices {
                // LU decomposition
                sgetrf_(_N, _N, ptr, _N, pivots, &error)
                if error < 0 {
                    throw NDArrayLUDecompError.IrregalValue
                } else if error > 0 {
                    throw NDArrayLUDecompError.SingularMatrix
                }
                
                // prod
                var magnitude: Float = 1
                var sign: Float = 1
                let p = pivots
                for i in 0..<size {
                    magnitude *= ptr.advanced(by: i*(size+1)).pointee
                    if (p+i).pointee != __CLPK_integer(i+1) {
                        sign *= -1
                    }
                }
                dst.pointee = sign * magnitude
                ptr += size*size
                dst += 1
            }
        }
    }
    
    return NDArray(shape: [Int](arg.shape.dropLast(2)),
                   elements: out)
}

/// Calculate matrix inverse.
///
/// If argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two dimensions.
public func inv(_ arg: NDArray) throws -> NDArray {
    precondition(arg.ndim > 1, "NDArray has shorter ndim(\(arg.ndim)) than 2.")
    let size = arg.shape[arg.ndim-1]
    precondition(arg.shape[arg.ndim-2] == size, "NDArray is not stack of square matrices: shape(\(arg.shape))")
    
    let volume = arg.volume
    var elements = gatherElements(arg)
    
    let numMatrices = volume / (size*size)
    
    var N = __CLPK_integer(size)
    let _N = UnsafeMutablePointer(&N)
    var pivots = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: size)
    var workspace = UnsafeMutablePointer<__CLPK_real>.allocate(capacity: size)
    var error: __CLPK_integer = 0
    
    defer {
        pivots.deallocate(capacity: size)
        workspace.deallocate(capacity: size)
    }
    
    try elements.withUnsafeMutablePointer { ptr in
        var ptr = ptr
        for _ in 0..<numMatrices {
            
            sgetrf_(_N, _N, ptr, _N, pivots, &error)
            if error < 0 {
                throw NDArrayLUDecompError.IrregalValue
            } else if error > 0 {
                throw NDArrayLUDecompError.SingularMatrix
            }
            
            sgetri_(_N, ptr, _N, pivots, workspace, _N, &error)
            if error < 0 {
                throw NDArrayInvError.IrregalValue
            } else if error > 0 {
                throw NDArrayInvError.SingularMatrix
            }
            
            ptr += size*size
        }
    }
    
    return NDArray(shape: arg.shape, elements: elements)
}

/// Calculate singular value decomposition.
///
/// If argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two dimensions.
/// `U` and `VT` are stack of matrices, and S is stack of vectors.
public func svd(_ arg: NDArray) throws -> (U: NDArray, S: NDArray, VT: NDArray) {
    precondition(arg.ndim > 1, "NDArray has shorter ndim(\(arg.ndim)) than 2.")
    let m = arg.shape[arg.ndim-2]
    let n = arg.shape[arg.ndim-1]
    
    var elements = gatherElements(arg.swapAxes(-1, -2))
    let outerShape = [Int](arg.shape.dropLast(2))
    let outerVolume = outerShape.prod()
    
    var jobz = Int8(UnicodeScalar("A")!.value)
    var _m = __CLPK_integer(m)
    let mp = UnsafeMutablePointer(&_m)
    var _n = __CLPK_integer(n)
    let np = UnsafeMutablePointer(&_n)
    let minMN = min(m, n)
    let lwork = 3*minMN*minMN + max(max(m, n), 4*minMN*minMN + 4*minMN)
    let work = UnsafeMutablePointer<Float>.allocate(capacity: lwork)
    var _lwork = __CLPK_integer(lwork)
    let iwork = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: 8*minMN)
    var info: __CLPK_integer = 0
    defer {
        work.deallocate(capacity: lwork)
        iwork.deallocate(capacity: 8*minMN)
    }
    
    var u = NDArrayData<Float>(size: outerVolume * m * m)
    var s = NDArrayData<Float>(size: outerVolume * minMN)
    var vt = NDArrayData<Float>(size: outerVolume * n * n)
    
    try elements.withUnsafeMutablePointer { ep in
        try u.withUnsafeMutablePointer { u in
            try s.withUnsafeMutablePointer { s in
                try vt.withUnsafeMutablePointer { vt in
                    var ep = ep
                    var u = u
                    var s = s
                    var vt = vt
                    
                    for _ in 0..<outerVolume {
                        sgesdd_(&jobz,
                                mp, np,
                                ep, mp,
                                s,
                                u, mp,
                                vt, np,
                                work,
                                &_lwork,
                                iwork,
                                &info)
                        
                        if info < 0 {
                            throw NDArraySVDError.IrregalValue
                        } else if info > 0 {
                            throw NDArraySVDError.NotConverge
                        }
                        
                        ep += m*n
                        u += m*m
                        s += minMN
                        vt += n*n
                    }
                }
            }
        }
    }
    
    return (U: NDArray(shape:outerShape + [m, m], elements: u).swapAxes(-1, -2),
            S: NDArray(shape: outerShape + [minMN], elements: s),
            VT: NDArray(shape: outerShape + [n, n], elements: vt).swapAxes(-1, -2))
}

public func pinv(_ arg: NDArray) throws -> NDArray {
    precondition(arg.ndim > 1, "NDArray has shorter ndim(\(arg.ndim)) than 2.")
    let m = arg.shape[arg.ndim-2]
    let n = arg.shape[arg.ndim-1]
    let outerShape = [Int](arg.shape.dropLast(2))
    
    let (u, s, vt) = try svd(arg)
    var S = NDArray.zeros(outerShape + [m, n])
    
    setSubarray(array: &S,
                indices: [NDArrayIndexElement?](repeating: nil, count: outerShape.count) + [..<min(m, n), ..<min(m, n)],
                newValue: NDArray.diagonal(1/s))
    
    return vt.swapAxes(-1, -2) |*| S |*| u.swapAxes(-1, -2)
}

public enum NDArrayLUDecompError: Error {
    case IrregalValue
    case SingularMatrix
}

public enum NDArrayInvError: Error {
    case IrregalValue
    case SingularMatrix
}

public enum NDArraySVDError: Error {
    case IrregalValue
    case NotConverge
}
