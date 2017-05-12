
extension NDArray {
    
    /// Create new NDArray which contains only specified rows.
    public func select(_ indices: [Int]) -> NDArray {
        return NDArray.stack(indices.map{ self[$0] })
    }
    
    /// Create new NDArray which contains only rows fufill `predicate`.
    public func select(where predicate: (NDArray)->Bool) -> NDArray {
        return NDArray.stack(filter(predicate))
    }
    
    /// Get row indices which fulfill `predicate`.
    public func indices(where predicate: (NDArray)->Bool) -> [Int] {
        return enumerated().flatMap { i, array in
            predicate(array) ? i : nil
        }
    }
    
}
