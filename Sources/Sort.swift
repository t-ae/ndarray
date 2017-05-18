
import Accelerate

/// Sort 1 dimensional NDArray.
public func sort(_ arg: NDArray, ascending: Bool = true) -> NDArray {
    precondition(arg.ndim == 1, "`arg` must be 1 dimensional.")
    var data = gatherElements(arg)
    data.withUnsafeMutablePointer { p in
        vDSP_vsort(p, vDSP_Length(data.count), ascending ? 1 : -1)
    }
    return NDArray(shape: [data.count], elements: data)
}

/// Index sort 1 dimensional NDArray.
/* Not working now
public func argsort(_ arg: NDArray, ascending: Bool = true) -> [UInt] {
    precondition(arg.ndim == 1, "`arg` must be 1 dimensional.")
    var index = [vDSP_Length](repeating: 0, count: arg.volume)
    let data = gatherElements(arg)
    vDSP_vsorti(data.pointer, &index, nil, vDSP_Length(index.count), ascending ? 1 : -1)
    return index
}
*/
