
import Foundation
import Accelerate

/// Calculate Frobenius norm
public func norm(_ arg: NDArray) -> Float {
    return sqrtf(sum(arg*arg).asScalar())
}

/// Calculate matrix inverse
///
/// If argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes
public func inv(_ arg: NDArray) throws -> NDArray {
    precondition(arg.ndim > 1)
    let size = arg.shape[arg.ndim-1]
    precondition(arg.shape[arg.ndim-2] == size)
    
    let volume = arg.volume
    var elements = gatherElements(arg, forceUniqueReference: true)
    
    let numMatrices = volume / (size*size)
    
    var N = __CLPK_integer(size)
    var pivots = UnsafeMutablePointer<__CLPK_integer>.allocate(capacity: size)
    var workspace = UnsafeMutablePointer<__CLPK_real>.allocate(capacity: size)
    var error: __CLPK_integer = 0
    
    defer {
        pivots.deallocate(capacity: size)
        workspace.deallocate(capacity: size)
    }
    
    var ptr = UnsafeMutablePointer(mutating: elements)
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
    
    return NDArray(shape: arg.shape, elements: elements)
}

public enum NDArrayInvError: Error {
    case IrregalValue
    case SingularMatrix
}
