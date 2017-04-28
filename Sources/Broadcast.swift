/// Broadcast two arrays.
func broadcast(_ lhs: NDArray, _ rhs: NDArray) -> (NDArray, NDArray) {
    if lhs.shape == rhs.shape {
        return (lhs, rhs)
    }
    
    var (lhs, rhs) = (lhs, rhs)
    
    let d = lhs.shape.count - rhs.shape.count
    if d < 0 {
        lhs.shape = [Int](repeating: 1, count: -d) + lhs.shape
        lhs.strides = [Int](repeating: 0, count: -d) + lhs.strides
    } else if d > 0 {
        rhs.shape = [Int](repeating: 1, count: d) + rhs.shape
        rhs.strides = [Int](repeating: 0, count: d) + rhs.strides
    }
    
    for i in 0..<lhs.ndim {
        if lhs.shape[i] == rhs.shape[i] {
            continue
        } else if(lhs.shape[i] == 1) {
            lhs.shape[i] = rhs.shape[i]
            lhs.strides[i] = 0
        } else if(rhs.shape[i] == 1) {
            rhs.shape[i] = lhs.shape[i]
            rhs.strides[i] = 0
        } else {
            preconditionFailure("Can't broadcast: \(lhs.shape) and \(rhs.shape)")
        }
    }
    
    return (lhs, rhs)
}

/// Broadcast arg to shape.
func broadcast(_ arg: NDArray, to shape: [Int]) -> NDArray {
    precondition(arg.shape.count <= shape.count, "Can't broadcast: \(arg.shape) to \(shape)")
    if arg.shape == shape {
        return arg
    }
    
    var arg = arg
    
    let d = shape.count - arg.shape.count
    arg.shape = [Int](repeating: 1, count: d) + arg.shape
    arg.strides = [Int](repeating: 0, count: d) + arg.strides
    
    for i in 0..<arg.ndim {
        if arg.shape[i] == shape[i] {
            continue
        } else if arg.shape[i] == 1 {
            arg.shape[i] = shape[i]
            arg.strides[i] = 0
        } else {
            preconditionFailure("Can't broadcast: \(arg.shape) to \(shape)")
        }
    }
    
    return arg
}
