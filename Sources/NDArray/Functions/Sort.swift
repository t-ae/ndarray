
import Accelerate

/// Sort elements along specified axis.
public func sort(_ arg: NDArray, along axis: Int = -1, ascending: Bool = true) -> NDArray {
    
    let axis = normalizeAxis(axis: axis, ndim: arg.ndim)
    
    let size = arg.shape[axis]
    
    let transposed = arg.moveAxis(from: axis, to: -1)
    
    var data = gatherElements(transposed)
    
    data.withUnsafeMutableBufferPointer {
        var p = $0.baseAddress!
        for _ in 0..<$0.count/size {
            vDSP_vsort(p, vDSP_Length(size), ascending ? 1 : -1)
            p += size
        }
    }
    return NDArray(shape: transposed.shape, elements: data).moveAxis(from: -1, to: axis)
}

/// Index sort 1 dimensional NDArray.
public func argsort(_ arg: NDArray, ascending: Bool = true) -> [UInt] {
    precondition(arg.ndim == 1, "`arg` must be 1 dimensional.")
    var index = [vDSP_Length](0..<vDSP_Length(arg.volume))
    let data = gatherElements(arg)
    data.withUnsafeBufferPointer {
        vDSP_vsorti($0.baseAddress!, &index, nil, vDSP_Length(index.count), ascending ? 1 : -1)
    }
    return index
}

