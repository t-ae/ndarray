
import Foundation

/// Calculate Frobenius norm
public func norm(_ arg: NDArray) -> Float {
    return sqrtf(sum(arg*arg).asScalar())
}
