
import Accelerate

extension NDArray {
    
    public static func stack(_ arrays: [NDArray], newAxis: Int = 0) -> NDArray {
        let shape = arrays.first!.shape
        precondition(arrays.map { $0.shape == shape }.all())
        
        let newAxis = normalizeAxis(axis: newAxis, ndim: shape.count+1)
        
        let reshaped = arrays.map { $0.reshaped($0.shape.inserting(1, at: newAxis)) }
        return concat(reshaped, along: newAxis)
    }
    
    public static func concat(_ arrays: [NDArray], along axis: Int) -> NDArray {
        let axis = normalizeAxis(axis: axis, ndim: arrays.first!.shape.count)
        
        let shapes = arrays.map { $0.shape.removing(at: axis) }
        let shape = shapes.first!
        precondition(shapes.map { $0 == shape }.all())
        
        let elementsList = arrays.map { gatherElements($0) }
        let majorShape = [Int](shape.prefix(axis))
        let minorShape = [Int](shape.dropFirst(axis))
        let blockSize = minorShape.reduce(1, *)
        
        let sizes = arrays.map { $0.shape[axis] }
        let newShape = shape.inserting(sizes.reduce(0, +), at: axis)
        let volume = newShape.reduce(1, *)
        
        var srcs = elementsList.map { UnsafePointer($0) }
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        var dstPtr = dst
        for _ in 0..<majorShape.reduce(1, *) {
            for i in 0..<srcs.count {
                let size = blockSize*sizes[i]
                cblas_scopy(Int32(size), srcs[i], 1, dstPtr, 1)
                srcs[i] += size
                dstPtr += size
            }
        }
        
        return NDArray(shape: newShape, elements: [Float](UnsafeBufferPointer(start: dst, count: volume)))
    }
}
