
extension NDArray {
    public init(scalar: Float) {
        self.init(shape: [], elements: [scalar])
    }
    
    /// check if this ndarray is scalar
    var isScalar: Bool {
        return self.shape.isEmpty
    }
    
    /// get scalar value
    func asScalar() -> Float {
        precondition(isScalar)
        return data[baseOffset]
    }
}
