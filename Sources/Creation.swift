import Foundation

extension NDArray {
    
    /// Create ndarray filled with specified values
    public static func filled(_ value: Float, shape: [Int]) -> NDArray {
        precondition(shape.map { $0 >= 0 }.all())
        return NDArray(shape: shape,
                       strides: [Int](repeating: 0, count: shape.count),
                       baseOffset: 0,
                       data: [value])
    }
    
    /// Create ndarray filled with 0s
    public static func zeros(_ shape: [Int]) -> NDArray {
        precondition(shape.map { $0 >= 0 }.all())
        return filled(0, shape: shape)
    }
    
    /// Create ndarray filled with 1s
    public static func ones(_ shape: [Int]) -> NDArray {
        precondition(shape.map { $0 >= 0 }.all())
        return filled(1, shape: shape)
    }
    
    /// Create identity matrix
    public static func eye(_ size: Int) -> NDArray {
        precondition(size > 0)
        let zeros = [Float](repeating: 0, count: size-1)
        let data = zeros + [1] + zeros
        return NDArray(shape: [size, size], strides: [-1, 1], baseOffset: size-1, data: data)
    }
    
    public static func diagonal(_ diag: [Float]) -> NDArray {
        return NDArray.eye(diag.count) * NDArray(diag)
    }
    
    public static func diagonal(_ diag: NDArray) -> NDArray {
        guard let size = diag.shape.last else {
            return diag
        }
        return NDArray.eye(size) <*> diag.reshaped(diag.shape + [1])
    }
    
    public static func range(_ count: Int) -> NDArray {
        precondition(count >= 0)
        return NDArray.range(0..<count)
    }
    
    public static func range(_ range: CountableRange<Int>) -> NDArray {
        let elements = range.map { Float($0) }
        return NDArray(elements)
    }
    
    public static func stride(from: Float, to: Float, by: Float = 1) -> NDArray {
        let elements = [Float](Swift.stride(from: from, to: to, by: by))
        return NDArray(elements)
    }
    
    public static func linspace(low: Float, high: Float, count: Int) -> NDArray {
        let elements = (0..<count).map { v -> Float in
            low + (high-low)*Float(v)/Float(count-1)
        }
        return NDArray(elements)
    }
}

