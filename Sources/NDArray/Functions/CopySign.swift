import Accelerate

/// Change the sign of `magnitude` to that of `sign`, element-wise.
public func copySign(magnitude: NDArray, sign: NDArray) -> NDArray {
    return apply(magnitude, sign, vvcopysignf)
}

/// Change the sign of `magnitude` to that of `sign`, element-wise.
public func copySign(magnitude: Float, sign: NDArray) -> NDArray {
    return apply(NDArray.filled(magnitude, shape: sign.shape), sign, vvcopysignf)
}
