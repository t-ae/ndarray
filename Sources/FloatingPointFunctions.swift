
import Accelerate

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

public func sinh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvsinhf)
}

public func cosh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvcoshf)
}

public func tanh(_ arg: NDArray) -> NDArray {
    return apply(arg, vvtanhf)
}

// MARK: Util
private typealias vvUnaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

private func apply(_ arg: NDArray, _ vvfunc: vvUnaryFunc) -> NDArray {
    let volume = arg.volume
    var count = Int32(volume)
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        let src = UnsafePointer(arg.data).advanced(by: arg.baseOffset)
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        vvfunc(dst, src, &count)
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    } else {
        let elements = gatherElements(arg)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        
        vvfunc(dst, elements, &count)
        return NDArray(shape: arg.shape,
                       elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
