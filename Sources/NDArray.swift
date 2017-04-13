public struct NDArray {
    /// shape
    public internal(set) var shape: [Int]
    /// source of elements
    var data: [Float]
    
    /// strides for each dimensions
    var strides: [Int]
    /// base offset of data
    var baseOffset: Int
    
    init(shape:[Int], strides: [Int], baseOffset: Int, data: [Float]) {
        self.shape = shape
        self.strides = strides
        self.data = data
        self.baseOffset = baseOffset
    }
    
    public init(shape: [Int], elements: [Float]) {
        
        precondition(shape.reduce(1, *) == elements.count)
        
        self.init(shape: shape, strides: continuousStrides(shape: shape), baseOffset: 0, data: elements)
    }
    
    func getElement(_ ndIndex: [Int]) -> Float {
        let ndIndex = normalizeIndex(shape: shape, ndIndex: ndIndex)
        let index = indexOffset(strides: strides, ndIndex: ndIndex) + baseOffset
        return data[index]
    }
    
    /// Number of dimensions
    public var ndim: Int {
        return shape.count
    }
    
    /// ndarray's volume
    /// - Returns: Number of elements
    public var volume: Int {
        return shape.reduce(1, *)
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
