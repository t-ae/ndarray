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
    
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
    defer { dst.deallocate(capacity: volume) }
    
    var _volume = Int32(volume)
    
    let lElements = gatherElements(lhs)
    let rElements = gatherElements(rhs)
    
    vvpowf(dst, rElements, lElements, &_volume)
    
    return NDArray(shape: lhs.shape,
                   elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
}
