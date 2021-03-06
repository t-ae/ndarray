import Foundation
import Accelerate

/// Calculate Frobenius norm of matrices.
///
/// - Parameters:
///   - arg: stack of matrices
///   - axes: axes which matrices lie
///   - keepDims: Squash dimensions or not
public func matrixNorm(_ arg: NDArray, axes: (Int, Int) = (-1, -2), keepDims: Bool = false) -> NDArray {
    let ax0 = normalizeAxis(axis: axes.0, ndim: arg.ndim)
    let ax1 = normalizeAxis(axis: axes.1, ndim: arg.ndim)
    
    precondition(ax0 != ax1, "Duplicate axes given.")
    
    let greaterAxis = max(ax0, ax1)
    let lesserAxis = min(ax0, ax1)
    
    let sumRow = reduce(arg, along: greaterAxis, vDSP_svesq)
    let ans = sqrt(sum(sumRow, along: lesserAxis, keepDims: keepDims))
    
    return keepDims ? ans.expandDims(greaterAxis) : ans
}

/// Calcurate eunclid norms of vectors.
///
/// - Parameters:
///   - arg: stack of vectors
///   - axis: axis which vectors lie
///   - keepDims: Squash dimension or not
public func vectorNorm(_ arg: NDArray, axis: Int = -1, keepDims: Bool = false) -> NDArray {
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
    
    var pivots = [__CLPK_integer](repeating: 0, count: size)
    
    var n = __CLPK_integer(size)
    let _n = UnsafeMutablePointer(&n)
    
    var info: __CLPK_integer = 0
    
    var out = [Float](repeating: 0, count: numMatrices)
    
    try elements.withUnsafeMutableBufferPointer {
        var ptr = $0.baseAddress!
        try out.withUnsafeMutableBufferPointer {
            var dst = $0.baseAddress!
            for _ in 0..<numMatrices {
                // LU decomposition
                sgetrf_(_n, _n, ptr, _n, &pivots, &info)
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
                    if p[i] != __CLPK_integer(i+1) {
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
    var pivots = [__CLPK_integer](repeating: 0, count: size)
    var workspace = [__CLPK_real](repeating: 0, count: size)
    var info: __CLPK_integer = 0
    
    try elements.withUnsafeMutableBufferPointer {
        var ptr = $0.baseAddress!
        for _ in 0..<numMatrices {
            
            sgetrf_(_n, _n, ptr, _n, &pivots, &info)
            if info < 0 {
                throw LinearAlgebraError.IrregalValue(func: "sgetrf_", nth: -Int(info))
            } else if info > 0 {
                throw LinearAlgebraError.SingularMatrix
            }
            
            sgetri_(_n, ptr, _n, &pivots, &workspace, _n, &info)
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
    var work = [Float](repeating: 0, count: lwork)
    var _lwork = __CLPK_integer(lwork)
    var iwork = [__CLPK_integer](repeating: 0, count: 8*minMN)
    var info: __CLPK_integer = 0
    
    var jobz: Int8
    var u: [Float]
    var s = [Float](repeating: 0, count: outerVolume * minMN)
    var vt: [Float]
    let ucols: Int
    let vtrows: Int
    var ldvt: __CLPK_integer
    
    if fullMatrices {
        jobz = Int8(UnicodeScalar("A")!.value)
        u = [Float](repeating: 0, count: outerVolume * m * m)
        vt = [Float](repeating: 0, count: outerVolume * n * n)
        ucols = m
        vtrows = n
        ldvt = __CLPK_integer(n)
    } else {
        jobz = Int8(UnicodeScalar("S")!.value)
        u = [Float](repeating: 0, count: outerVolume * minMN * m)
        vt = [Float](repeating: 0, count: outerVolume * minMN * n)
        ucols = minMN
        vtrows = minMN
        ldvt = __CLPK_integer(minMN)
    }
    
    try elements.withUnsafeMutableBufferPointer {
        var ep = $0.baseAddress!
        try u.withUnsafeMutableBufferPointer {
            var u = $0.baseAddress!
            try s.withUnsafeMutableBufferPointer {
                var s = $0.baseAddress!
                try vt.withUnsafeMutableBufferPointer {
                    var vt = $0.baseAddress!
                    
                    for _ in 0..<outerVolume {
                        sgesdd_(&jobz, &_m, &_n,
                                ep, &lda,
                                s,
                                u, &ldu,
                                vt, &ldvt,
                                &work, &_lwork, &iwork, &info)
                        
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

/// Compute the rank of matrix
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
    var work = [Float](repeating: 0, count: lwork)
    var _lwork = __CLPK_integer(lwork)
    var iwork = [__CLPK_integer](repeating: 0, count: 8*min(m, n))
    var info: __CLPK_integer = 0
    var s = [Float](repeating: 0, count: min(m, n))
    
    a.withUnsafeMutableBufferPointer {
        let a = $0.baseAddress!
        s.withUnsafeMutableBufferPointer {
            let s = $0.baseAddress!
            sgesdd_(&jobz, &_n, &_m,
                    a, &lda,
                    s,
                    _dummy, _lddummy,
                    _dummy, _lddummy,
                    &work,  &_lwork, &iwork, &info)
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

