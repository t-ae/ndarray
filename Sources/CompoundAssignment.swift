
// MARK: - Scalar
public func +=(lhs: inout NDArray, rhs: Float) {
    lhs = lhs + rhs
}

public func -=(lhs: inout NDArray, rhs: Float) {
    lhs = lhs - rhs
}

public func *=(lhs: inout NDArray, rhs: Float) {
    lhs = lhs * rhs
}

public func /=(lhs: inout NDArray, rhs: Float) {
    lhs = lhs / rhs
}

// MARK: - NDArray
public func +=(lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs + rhs
}

public func -=(lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs - rhs
}

public func *=(lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs * rhs
}

public func /=(lhs: inout NDArray, rhs: NDArray) {
    lhs = lhs / rhs
}
