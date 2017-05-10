
import Accelerate

public func copySign(magnitude: NDArray, sign: NDArray) -> NDArray {
    return apply(magnitude, sign, vvcopysignf)
}
