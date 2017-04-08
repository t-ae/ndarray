
extension NDArray {
    public init(scalar: Float) {
        self.init(shape: [], elements: [scalar])
    }
    
    /// check if this ndarray is scalar
    var isScalar: Bool {
        return self.shape.isEmpty
    }
    
    /// get scalar value
    var scalar: Float {
        precondition(isScalar)
        return data[baseOffset]
    }
}

extension NDArray: ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral {
    public typealias IntegerLiteralType = Int
    public init(integerLiteral value: Int) {
        self.init(shape: [], elements: [Float(value)])
    }
    
    public typealias FloatLiteralType = Float
    public init(floatLiteral value: Float) {
        self.init(shape: [], elements: [value])
    }
}
