
import Foundation
import Accelerate

/// Calculate Frobenius norm.
public func matrixNorm(_ arg: NDArray, axes: (Int, Int) = (-1, -2), keepDims: Bool = false) -> NDArray {
    let axes = (normalizeAxis(axis: axes.0, ndim: arg.ndim),
                normalizeAxis(axis: axes.1, ndim: arg.ndim))
    
    precondition(axes.0 != axes.1, "Duplicate axes given")
    let sumRow = reduce(arg, along: axes.0, vDSP_svesq).expandDims(axes.0)
    let ans = sqrt(sum(sumRow, along: axes.1, keepDims: true))
    
    return keepDims ? ans : ans.squeezed()
}

/// Calcurate vector norms along axis.
public func vectorNorm(_ arg: NDArray, along axis: Int = -1, keepDims: Bool = false) -> NDArray {
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
    
    var n = __CLPK_integer(size)
    let _n = UnsafeMutablePointer(&n)
    
    var info: __CLPK_integer = 0
    
    var out = NDArrayData<Float>(size: numMatrices)
    
    try elements.withUnsafeMutablePointer { ptr in
        try out.withUnsafeMutablePointer { dst in
            var ptr = ptr
            var dst = dst
            for _ in 0..<numMatrices {
                // LU decomposition
                sgetrf_(_n, _n, ptr, _n, pivots, &info)
                if info < 0 {
                    throw LinearAlgebraError.IrregalValue(func: "sgetrf_", nth: -Int(info))
                } else if info > 0 {
                    throw LinearAlgebraError.SingularMatrix
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
    
    var n = __CLPK_integer(size)
    let _n = UnsafeMutablePointer(&n)
    var pivots = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: size)
    var workspace = UnsafeMutablePointer<__CLPK_real>.allocate(capacity: size)
    var info: __CLPK_integer = 0
    
    defer {
        pivots.deallocate(capacity: size)
        workspace.deallocate(capacity: size)
    }
    
    try elements.withUnsafeMutablePointer { ptr in
        var ptr = ptr
        for _ in 0..<numMatrices {
            
            sgetrf_(_n, _n, ptr, _n, pivots, &info)
            if info < 0 {
                throw LinearAlgebraError.IrregalValue(func: "sgetrf_", nth: -Int(info))
            } else if info > 0 {
                throw LinearAlgebraError.SingularMatrix
            }
            
            sgetri_(_n, ptr, _n, pivots, workspace, _n, &info)
            if info < 0 {
                throw LinearAlgebraError.IrregalValue(func: "sgetri_", nth: -Int(info))
            } else if info > 0 {
                throw LinearAlgebraError.SingularMatrix
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
public func svd(_ arg: NDArray, fullMatrices: Bool = true) throws -> (U: NDArray, S: NDArray, VT: NDArray) {
    precondition(arg.ndim > 1, "NDArray has shorter ndim(\(arg.ndim)) than 2.")
    let m = arg.shape[arg.ndim-2]
    let n = arg.shape[arg.ndim-1]
    
    var elements = gatherElements(arg.swapAxes(-1, -2)) // col major
    let outerShape = [Int](arg.shape.dropLast(2))
    let outerVolume = outerShape.prod()
    
    var _m = __CLPK_integer(m)
    var _n = __CLPK_integer(n)
    var lda = _m
    var ldu = _m
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
    
    var jobz: Int8
    var u: NDArrayData<Float>
    var s = NDArrayData<Float>(size: outerVolume * minMN)
    var vt: NDArrayData<Float>
    let ucols: Int
    let vtrows: Int
    var ldvt: __CLPK_integer
    
    if fullMatrices {
        jobz = Int8(UnicodeScalar("A")!.value)
        u = NDArrayData<Float>(size: outerVolume * m * m)
        vt = NDArrayData<Float>(size: outerVolume * n * n)
        ucols = m
        vtrows = n
        ldvt = __CLPK_integer(n)
    } else {
        jobz = Int8(UnicodeScalar("S")!.value)
        u = NDArrayData<Float>(size: outerVolume * minMN * m)
        vt = NDArrayData<Float>(size: outerVolume * minMN * n)
         vt = NDArrayData<Float>(value: 0, count: outerVolume * minMN * n)
        ucols = minMN
        vtrows = minMN
        ldvt = __CLPK_integer(minMN)
    }
    
    try elements.withUnsafeMutablePointer { ep in
        try u.withUnsafeMutablePointer { u in
            try s.withUnsafeMutablePointer { s in
                try vt.withUnsafeMutablePointer { vt in
                    var ep = ep
                    var u = u
                    var s = s
                    var vt = vt
                    
                    for _ in 0..<outerVolume {
                        sgesdd_(&jobz, &_m, &_n,
                                ep, &lda,
                                s,
                                u, &ldu,
                                vt, &ldvt,
                                work, &_lwork, iwork, &info)
                        
                        if info < 0 {
                            throw LinearAlgebraError.IrregalValue(func: "sgesdd_", nth: -Int(info))
                        } else if info > 0 {
                            throw LinearAlgebraError.NotConverged
                        }
                        
                        ep += m*n
                        u += m*ucols
                        s += minMN
                        vt += vtrows*n
                    }
                }
            }
        }
    }
    
    return (U: NDArray(shape:outerShape + [ucols, m], elements: u).swapAxes(-1, -2),
            S: NDArray(shape: outerShape + [minMN], elements: s),
            VT: NDArray(shape: outerShape + [n, vtrows], elements: vt).swapAxes(-1, -2))
}

/// Calculate pseudo inverse of matrix.
public func pinv(_ arg: NDArray, rcond: Float = 1e-5) throws -> NDArray {
    precondition(arg.ndim == 2, "NDArray must be a matrix.")

    var (u, s, vt) = try svd(arg, fullMatrices: false)
    
    let cutoff = rcond*max(s, along: -1).asScalar()
    
    for i in 0..<s.shape[0] {
        if s[i].asScalar() > cutoff {
            s[i] = 1/s[i]
        } else {
            s[i] = NDArray(scalar: 0)
        }
    }
    
    return vt.swapAxes(-1, -2) |*| (s.expandDims(-1) * u.swapAxes(-1, -2))
}

// Compute the rank of matrix
public func matrixRank(_ arg: NDArray, tol: Float? = nil) -> Int {
    precondition(arg.ndim == 2, "NDArray must be a matrix.")
    let m = arg.shape[arg.ndim-2]
    let n = arg.shape[arg.ndim-1]
    
    if m < 2 || n < 2 {
        return arg.data.filter { $0 != 0 }.isEmpty ? 0 : 1
    }
    
    // treat as n by m matrix
    var jobz = Int8(UnicodeScalar("N").value)
    var _m = __CLPK_integer(m)
    var _n =  __CLPK_integer(n)
    
    var a = gatherElements(arg)
    var lda = _n
    
    var dummy: Float = 0
    var lddummy: __CLPK_integer = 1
    let _dummy = UnsafeMutablePointer(&dummy)
    let _lddummy = UnsafeMutablePointer(&lddummy)
    
    let lwork = 3*min(m, n) + 2*max(max(m, n), 6*min(m, n)) // not optimal size
    let work = UnsafeMutablePointer<Float>.allocate(capacity: lwork)
    var _lwork = __CLPK_integer(lwork)
    let iwork = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: 8*min(m, n))
    var info: __CLPK_integer = 0
    var s = NDArrayData<Float>(size: min(m, n))
    
    defer {
        work.deallocate(capacity: lwork)
        iwork.deallocate(capacity: 8*min(m, n))
    }
    
    a.withUnsafeMutablePointer { a in
        s.withUnsafeMutablePointer { s -> Void in
            sgesdd_(&jobz, &_n, &_m,
                    a, &lda,
                    s,
                    _dummy, _lddummy,
                    _dummy, _lddummy,
                    work,  &_lwork, iwork, &info)
        }
    }
    assert(info == 0)
    let maxS = s[0]
    
    let eps: Float = 1.1920929e-07 // np.finfo(np.float32).eps
    let tol = tol ?? maxS * Float(max(m, n)) * eps
    return s.prefix { $0 > tol }.count
}

public enum LinearAlgebraError: Error {
    case IrregalValue(func: String, nth: Int)
    case SingularMatrix
    case NotConverged
}

