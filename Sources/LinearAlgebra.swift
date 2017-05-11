
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

/// Calculate matrix inverse.
///
/// If argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes
public func inv(_ arg: NDArray) throws -> NDArray {
    precondition(arg.ndim > 1, "NDArray has shorter ndim(\(arg.ndim)) than 2.")
    let size = arg.shape[arg.ndim-1]
    precondition(arg.shape[arg.ndim-2] == size, "NDArray is not stack of matrices: shape(\(arg.shape))")
    
    let volume = arg.volume
    var elements = gatherElements(arg)
    
    let numMatrices = volume / (size*size)
    
    var N = __CLPK_integer(size)
    var pivots = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: size)
    var workspace = UnsafeMutablePointer<__CLPK_real>.allocate(capacity: size)
    var error: __CLPK_integer = 0
    
    defer {
        pivots.deallocate(capacity: size)
        workspace.deallocate(capacity: size)
    }
    
    // Force CoW
    try elements.withUnsafeMutableBufferPointer { p -> Void in
        var ptr = p.baseAddress!
        for _ in 0..<numMatrices {
            
            sgetrf_(&N, &N, ptr, &N, pivots, &error)
            if error < 0 {
                throw NDArrayInvError.IrregalValue
            } else if error > 0 {
                throw NDArrayInvError.SingularMatrix
            }
            
            sgetri_(&N, ptr, &N, pivots, workspace, &N, &error)
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

public enum NDArrayInvError: Error {
    case IrregalValue
    case SingularMatrix
}
