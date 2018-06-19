import Foundation

/// Calculate covariance matrix.
public func cov(_ arg: NDArray) -> NDArray {
    var arg = arg
    switch arg.ndim {
    case 1:
        arg = arg.reshaped(arg.shape.inserting(1, at: 0))
    case 2:
        break
    default:
        fatalError("`cov` supports only 1D or 2D array.")
    }
    
    let msub = arg - mean(arg, along: 1, keepDims: true)
    
    return mean(msub.expandDims(0) * msub.expandDims(1), along: -1)
}
