extension NDArray {
    
    /// Get dimensions reversed ndarray
    public func transposed() -> NDArray {
        let axes = [Int]((0..<shape.count).reversed())
        return transposed(axes: axes)
    }
    
    /// Get dimensions permutad ndarray
    public func transposed(axes: [Int]) -> NDArray {
        precondition(axes.sorted() == [Int](0..<self.ndim))
        var x = self
        for (i, ax) in axes.enumerated() {
            x.strides[i] = self.strides[ax]
            x.shape[i] = self.shape[ax]
        }
        return x
    }
    
    /// Get reshaped ndarray
    public func reshaped(_ shape: [Int]) -> NDArray {
        
        var shape = shape
        if let arbit = shape.index(of: -1) {
            shape[arbit] = self.volume / shape.removed(at: arbit).reduce(1, *)
        }
        
        precondition(shape.map { $0 >= 0 }.all())
        precondition(self.volume == shape.reduce(1, *))
        
        if isNormalized {
            return NDArray(shape: shape, elements: data)
        } else {
            let elements = gatherElements(self)
            return NDArray(shape: shape, elements: elements)
        }
    }
}
