
extension NDArray {
    public init(scalar: Float) {
        self.init(shape: [], elements: [scalar])
    }
    
    /// check if this ndarray is scalar
    public var isScalar: Bool {
        return self.shape.isEmpty
    }
    
    /// get scalar value
    public func asScalar() -> Float {
        precondition(isScalar)
        return data[baseOffset]
    }
}
