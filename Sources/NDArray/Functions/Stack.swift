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
        
        var dst = [Float](repeating: 0, count: volume)
        
        var start = 0
        for m in 0..<majorShape.prod() {
            for (e, size) in zip(elementsList, sizes) {
                dst.replaceSubrange(start..<start+size*blockSize, with: e[m*size*blockSize..<(m+1)*size*blockSize])
                start += size*blockSize
            }
        }
        
        return NDArray(shape: newShape, elements: dst)
    }
}
