/// Broadcast two arrays.
func broadcast(_ lhs: NDArray, _ rhs: NDArray) -> (NDArray, NDArray) {
    if lhs.shape == rhs.shape {
        return (lhs, rhs)
    }
    
    var (lShape, rShape) = (lhs.shape, rhs.shape)
    var (lStrides, rStrides) = (lhs.strides, rhs.strides)
    
    let d = lShape.count - rShape.count
    if d < 0 {
        lShape = [Int](repeating: 1, count: -d) + lShape
        lStrides = [Int](repeating: 0, count: -d) + lStrides
    } else if d > 0 {
        rShape = [Int](repeating: 1, count: d) + rShape
        rStrides = [Int](repeating: 0, count: d) + rStrides
    }
    
    for i in 0..<lShape.count {
        if lShape[i] == rShape[i] {
            continue
        } else if(lShape[i] == 1) {
            lShape[i] = rShape[i]
            lStrides[i] = 0
        } else if(rShape[i] == 1) {
            rShape[i] = lShape[i]
            rStrides[i] = 0
        } else {
            preconditionFailure()
        }
    }
    
    let lArray = NDArray(shape: lShape,
                         strides: lStrides,
                         baseOffset: lhs.baseOffset,
                         data: lhs.data)
    let rArray = NDArray(shape: rShape,
                         strides: rStrides,
                         baseOffset: rhs.baseOffset,
                         data: rhs.data)
    
    return (lArray, rArray)
}

/// Broadcast arg to shape.
func broadcast(_ arg: NDArray, to shape: [Int]) -> NDArray {
    precondition(arg.shape.count <= shape.count)
    if arg.shape == shape {
        return arg
    }
    
    let d = shape.count - arg.shape.count
    var newShape = [Int](repeating: 1, count: d) + arg.shape
    var newStrides = [Int](repeating: 0, count: d) + arg.strides
    
    for i in 0..<newShape.count {
        if newShape[i] == shape[i] {
            continue
        } else if newShape[i] == 1 {
            newShape[i] = shape[i]
            newStrides[i] = 0
        } else {
            preconditionFailure()
        }
    }
    
    return NDArray(shape: newShape,
                   strides: newStrides,
                   baseOffset: arg.baseOffset,
                   data: arg.data)
    
}
