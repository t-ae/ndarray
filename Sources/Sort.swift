
import Accelerate

/// Sort elements along specified axis.
public func sort(_ arg: NDArray, along axis: Int = -1, ascending: Bool = true) -> NDArray {
    
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let size = arg.shape[axis]
    
    let transposed = arg.moveAxis(from: axis, to: -1)
    
    var data = gatherElements(transposed)
    
    data.withUnsafeMutablePointer { p in
        var p = p
        for _ in 0..<data.count/size {
            vDSP_vsort(p, vDSP_Length(size), ascending ? 1 : -1)
            p += size
        }
    }
    return NDArray(shape: transposed.shape, elements: data).moveAxis(from: -1, to: axis)
}

/// Index sort 1 dimensional NDArray.
public func argsort(_ arg: NDArray, ascending: Bool = true) -> [Int] {
    precondition(arg.ndim == 1)
    if ascending {
        return arg.enumerated()
            .sorted { l, r in l.element.asScalar() < r.element.asScalar() }
            .map { $0.offset }
    } else {
        return arg.enumerated()
            .sorted { l, r in l.element.asScalar() >= r.element.asScalar() }
            .map { $0.offset }
    }
}

/* This may faster, but somehow not works
public func argsort(_ arg: NDArray, ascending: Bool = true) -> [UInt] {
    precondition(arg.ndim == 1, "`arg` must be 1 dimensional.")
    var index = [vDSP_Length](repeating: 0, count: arg.volume)
    let data = gatherElements(arg)
    vDSP_vsorti(data.pointer, &index, nil, vDSP_Length(index.count), ascending ? 1 : -1)
    return index
}
*/
