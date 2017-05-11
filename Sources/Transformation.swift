extension NDArray {
    
    /// Get dimensions reversed NDArray.
    public func transposed() -> NDArray {
        let axes = [Int]((0..<ndim).reversed())
        return transposed(axes)
    }
    
    /// Get dimensions permuted NDArray.
    public func transposed(_ axes: [Int]) -> NDArray {
        var axes = axes
        for i in 0..<axes.count {
            if axes[i] < 0 {
                axes[i] += shape[i]
            }
        }
        precondition(axes.sorted() == [Int](0..<ndim), "Axes(\(axes)) don't match NDArray(shape: \(shape))")
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
            shape[arbit] = volume / shape.removing(at: arbit).prod()
        }
        
        precondition(shape.all { $0 >= 0 }, "Invalid shape.")
        precondition(volume == shape.prod(), "New shape's volume must match with the number of elements.")
        
        let elements = gatherElements(self)
        return NDArray(shape: shape, elements: elements)
    }
    
    /// Get raveled NDArray.
    public func raveled() -> NDArray {
        return reshaped([-1])
    }
    
    /// Remove size 1 dimensions.
    public func squeeze() -> NDArray {
        var x = self
        x.shape = shape.filter { $0 != 1 }
        x.strides = zip(strides, shape).flatMap { stride, size in
            size == 1 ? nil : stride
        }
        return x
    }
    
    /// Insert a new axis, corresponding to a given position in the array shape.
    public func expandDims(_ newAxis: Int) -> NDArray {
        var x = self
        let newAxis = normalizeAxis(axis: newAxis, ndim: ndim+1)
        x.shape = shape.inserting(1, at: newAxis)
        x.strides = strides.inserting(0, at: newAxis)
        return x
    }
    
    /// Flip axis
    public func flipped(_ axis: Int) -> NDArray {
        
        let axis = normalizeAxis(axis: axis, ndim: ndim)
        
        var x = self
        x.baseOffset += x.strides[axis] * (x.shape[axis] - 1)
        x.strides[axis] = -x.strides[axis]
        
        return x
    }
}
