import Foundation

extension NDArray {
    
    /// Create NDArray filled with specified values.
    public static func filled(_ value: Float, shape: [Int]) -> NDArray {
        precondition(shape.all { $0 >= 0 }, "Shape(\(shape)) contains minus value.")
        return NDArray(shape: shape, elements: NDArrayData(value: value, count: shape.prod()))
    }
    
    /// Create NDArray filled with 0s.
    public static func zeros(_ shape: [Int]) -> NDArray {
        return filled(0, shape: shape)
    }
    
    /// Create NDArray filled with 1s.
    public static func ones(_ shape: [Int]) -> NDArray {
        return filled(1, shape: shape)
    }
    
    /// Create uninitialized NDArray
    public static func empty(_ shape: [Int]) -> NDArray {
        precondition(shape.all { $0 >= 0 }, "Shape(\(shape)) contains minus value.")
        
        return NDArray(shape: shape, elements: NDArrayData(size: shape.prod()))
    }
    
    /// Create identity matrix.
    public static func eye(_ size: Int) -> NDArray {
        precondition(size >= 0, "Size(\(size)) must >= 0.")
        var data = NDArrayData(value: 0, count: size*size)
        data.withUnsafeMutablePointer { p in
            var p = p
            for _ in 0..<size {
                p.pointee = 1
                p += size + 1
            }
        }
        return NDArray(shape: [size, size], elements: data)
    }
    
    /// Create diagonal matrix.
    public static func diagonal(_ diag: [Float]) -> NDArray {
        return NDArray.eye(diag.count) * NDArray(diag)
    }
    
    /// Create diagonal matrix.
    ///
    /// If the argument is N-D, N > 1, it treated as stack of vectors
    /// and result is (N+1)-D array.
    public static func diagonal(_ diag: NDArray) -> NDArray {
        guard let size = diag.shape.last else {
            return diag
        }
        return NDArray.eye(size) * diag.expandDims(-1)
    }
    
    /// Create contiguous NDArray 0..<count.
    public static func range(_ count: Int) -> NDArray {
        precondition(count >= 0, "Count(\(count)) must >= 0.")
        return NDArray.range(0..<count)
    }
    
    /// Create contiguous NDArray.
    public static func range(_ range: CountableRange<Int>) -> NDArray {
        var elements = NDArrayData(size: range.count)
        elements.withUnsafeMutablePointer { p in
            var p = p
            for e in range {
                p.pointee = Float(e)
                p += 1
            }
        }
        return NDArray(shape: [range.count], elements: elements)
    }
    
    /// Create evenly spaced NDArray.
    public static func stride(from: Float, to: Float, by: Float = 1) -> NDArray {
        let elements = [Float](Swift.stride(from: from, to: to, by: by))
        return NDArray(elements)
    }
    
    /// Create evenly spaced NDArray.
    ///
    /// `count` elements are in the interval [low, high].
    public static func linspace(low: Float, high: Float, count: Int) -> NDArray {
        var elements = NDArrayData(size: count)
        elements.withUnsafeMutablePointer { p in
            var p = p
            for v in 0..<count {
                p.pointee = low + (high-low)*Float(v)/Float(count-1)
                p += 1
            }
        }
        return NDArray(shape: [count], elements: elements)
    }
}

