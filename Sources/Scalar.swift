
extension NDArray {
    /// Create scalar NDArray.
    public init(scalar: Float) {
        self.init(shape: [], elements: [scalar])
    }
    
    /// Check if this ndarray is scalar.
    public var isScalar: Bool {
        return self.shape.isEmpty
    }
    
    /// Get scalar value.
    public func asScalar() -> Float {
        precondition(isScalar)
        return data[baseOffset]
    }
}
