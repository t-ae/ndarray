import Accelerate

precedencegroup PowPrecedense {
    associativity: left
    higherThan: MultiplicationPrecedence
}

infix operator **: PowPrecedense

public func **(lhs: NDArray, rhs: Float) -> NDArray {
    return lhs ** NDArray(scalar: rhs)
}

public func **(lhs: Float, rhs: NDArray) -> NDArray {
    return NDArray(scalar: lhs) ** rhs
}

public func **(lhs: NDArray, rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vvpowf)
}
