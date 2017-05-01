public struct NDArray {
    /// Shape of NDArray.
    public internal(set) var shape: [Int]
    /// Source of elements.
    public internal(set) var data: [Float]
    
    /// Strides for each dimensions.
    public internal(set) var strides: [Int]
    /// Base offset of data.
    public internal(set) var baseOffset: Int
    
    init(shape:[Int], strides: [Int], baseOffset: Int, data: [Float]) {
        assert(shape.count == strides.count)
        assert(shape.all { $0 >= 0 })
        assert(0 <= baseOffset && (baseOffset < data.count || data.isEmpty))
        self.shape = shape
        self.strides = strides
        self.data = data
        self.baseOffset = baseOffset
    }
    
    /// Init with contiguous strides.
    public init(shape: [Int], elements: [Float]) {
        precondition(shape.all { $0 >= 0 }, "Shape(\(shape)) contains minus value.")
        precondition(shape.prod() == elements.count, "Elements count must correspond to product of shape.")
        
        self.init(shape: shape,
                  strides: contiguousStrides(shape: shape),
                  baseOffset: 0,
                  data: elements)
    }
    
    /// Get all elements.
    public func elements() -> [Float] {
        return gatherElements(self)
    }
    
    /// Get single element.
    public func element(at ndIndex: [Int]) -> Float {
        precondition(ndIndex.count == strides.count, "Invalid index for single element.")
        let ndIndex = normalizeIndex(shape: shape, ndIndex: ndIndex)
        let index = indexOffset(strides: strides, ndIndex: ndIndex) + baseOffset
        return data[index]
    }
    
    /// Number of dimensions.
    public var ndim: Int {
        return shape.count
    }
    
    /// NDArray's volume.
    /// - Returns: Number of elements
    public var volume: Int {
        return shape.prod()
    }
    
    public func asContiguousArray() -> NDArray {
        return NDArray(shape: shape, elements: elements())
    }
}

extension NDArray: Equatable {
    
}

public func ==(lhs: NDArray, rhs: NDArray) -> Bool {
    guard lhs.shape == rhs.shape else {
        return false
    }
    return gatherElements(lhs) == gatherElements(rhs)
}
