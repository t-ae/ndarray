import Accelerate

precedencegroup PowPrecedense {
    associativity: left
    higherThan: MultiplicationPrecedence
}

infix operator **: PowPrecedense

public func **(lhs: NDArray, rhs: Float) -> NDArray {
    let f: vvUnaryFunc = {
        var rhs = rhs
        vvpowsf($0, &rhs, $1, $2)
    }
    return apply(lhs, f)
}

public func **(lhs: Float, rhs: NDArray) -> NDArray {
    return NDArray(scalar: lhs) ** rhs
}

public func **(lhs: NDArray, rhs: NDArray) -> NDArray {
    return apply(rhs, lhs, vvpowf)
}
