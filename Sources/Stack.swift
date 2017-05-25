
import Accelerate

extension NDArray {
    
    /// Concatenate NDArrays.
    ///
    /// All arrays must have same shape.
    public static func stack(_ arrays: [NDArray], newAxis: Int = 0) -> NDArray {
        let shape = arrays.first!.shape
        precondition(arrays.all { $0.shape == shape }, "All NDArrays must have same shape.")
        
        let reshaped = arrays.map { $0.expandDims(newAxis) }
        return concat(reshaped, along: newAxis)
    }
    
    /// Concatenate NDArrays along a given axis.
    ///
    /// All arrays must have same shape without specified axis.
    public static func concat(_ arrays: [NDArray], along axis: Int) -> NDArray {
        let axis = normalizeAxis(axis: axis, ndim: arrays.first!.shape.count)
        
        let shapes = arrays.map { $0.shape.removing(at: axis) }
        let shape = shapes.first!
        precondition(shapes.all { $0 == shape },
                     "All NDArray dimensions except for the concatenation axis must match exactly.")
        
        let elementsList = arrays.map { gatherElements($0) }
        let majorShape = [Int](shape.prefix(axis))
        let minorShape = [Int](shape.dropFirst(axis))
        let blockSize = minorShape.prod()
        
        let sizes = arrays.map { $0.shape[axis] }
        let newShape = shape.inserting(sizes.sum(), at: axis)
        let volume = newShape.prod()
        
        var srcs = elementsList.map { $0.pointer }
        
        var dst = NDArrayData<Float>(size: volume)
        
        dst.withUnsafeMutablePointer {
            var dstPtr = $0
            for _ in 0..<majorShape.prod() {
                for i in 0..<srcs.count {
                    let size = blockSize*sizes[i]
                    cblas_scopy(Int32(size), srcs[i], 1, dstPtr, 1)
                    srcs[i] += size
                    dstPtr += size
                }
            }
        }
        
        return NDArray(shape: newShape, elements: dst)
    }
}
