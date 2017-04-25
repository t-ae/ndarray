extension NDArray {
    
    /// Get dimensions reversed NDArray.
    public func transposed() -> NDArray {
        let axes = [Int]((0..<ndim).reversed())
        return transposed(axes)
    }
    
    /// Get dimensions permuted NDArray.
    public func transposed(_ axes: [Int]) -> NDArray {
        precondition(axes.sorted() == [Int](0..<ndim))
        var x = self
        for (i, ax) in axes.enumerated() {
            x.strides[i] = self.strides[ax]
            x.shape[i] = self.shape[ax]
        }
        return x
    }
    
    /// Get reshaped NDArray.
    public func reshaped(_ shape: [Int]) -> NDArray {
        
        var shape = shape
        if let arbit = shape.index(of: -1) {
            shape[arbit] = self.volume / shape.removing(at: arbit).prod()
        }
        
        precondition(shape.map { $0 >= 0 }.all())
        precondition(self.volume == shape.prod())
        
        let elements = gatherElements(self)
        return NDArray(shape: shape, elements: elements)
    }
    
    /// Get raveled NDArray.
    public func raveled() -> NDArray {
        return reshaped([-1])
    }
}
