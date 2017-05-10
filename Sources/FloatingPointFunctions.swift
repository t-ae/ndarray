
import Accelerate

public func floor(_ arg: NDArray) -> NDArray {
    return apply(arg, vvfloorf)
}

public func ceil(_ arg: NDArray) -> NDArray {
    return apply(arg, vvceilf)
}

public func round(_ arg: NDArray) -> NDArray {
    return apply(arg, vvnintf)
}

public func abs(_ arg: NDArray) -> NDArray {
    return apply(arg, vDSP_vabs)
}

public func sqrt(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsqrtf)
}

public func log(_ arg: NDArray) -> NDArray {
    return apply(arg, vvlogf)
}

public func log2(_ arg: NDArray) -> NDArray {
    return apply(arg, vvlog2f)
}

public func log10(_ arg: NDArray) -> NDArray {
    return apply(arg, vvlog10f)
}

public func exp(_ arg: NDArray) -> NDArray {
    return apply(arg, vvexpf)
}

public func exp2(_ arg: NDArray) -> NDArray {
    return apply(arg, vvexp2f)
}

public func sin(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsinf)
}

public func cos(_ arg: NDArray) -> NDArray {
    return apply(arg, vvcosf)
}

public func tan(_ arg: NDArray) -> NDArray {
    return apply(arg, vvtanf)
}

public func asin(_ arg: NDArray) -> NDArray {
    return apply(arg, vvasinf)
}

public func acos(_ arg: NDArray) -> NDArray {
    return apply(arg, vvacosf)
}

public func atan(_ arg: NDArray) -> NDArray {
    return apply(arg, vvatanf)
}

public func sinh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsinhf)
}

public func cosh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvcoshf)
}

public func tanh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvtanhf)
}
