
extension NDArray {
    
    /// Create new NDArray which contains only specified rows.
    public func select(_ indices: [Int]) -> NDArray {
        guard !indices.isEmpty else {
            return NDArray.empty(shape.removing(at: 0).inserting(0, at: 0))
        }
        return NDArray.stack(indices.map{ self[$0] })
    }
    
    /// Create new NDArray which contains only rows fufill `predicate`.
    public func select(where predicate: (NDArray)->Bool) -> NDArray {
        let filtered = filter(predicate)
        guard !filtered.isEmpty else {
            return NDArray.empty(shape.removing(at: 0).inserting(0, at: 0))
        }
        return NDArray.stack(filtered)
    }
    
    /// Create new NDArray which contains only rows fufill `predicate` with index.
    public func select(where predicate: (Int, NDArray)->Bool) -> NDArray {
        let filtered = enumerated().flatMap { predicate($0, $1) ? $1 : nil }
        guard !filtered.isEmpty else {
            return NDArray.empty(shape.removing(at: 0).inserting(0, at: 0))
        }
        return NDArray.stack(filtered)
    }
    
    /// Create new NDArray with rows `mask == true`
    public func select(_ mask: [Bool]) -> NDArray {
        precondition(mask.count == shape[0])
        guard mask.contains(true) else {
            return NDArray.empty(shape.removing(at: 0).inserting(0, at: 0))
        }
        return NDArray.stack(zip(self, mask).flatMap { $1 ? $0 : nil })
    }
    
    /// Get row indices which fulfill `predicate`.
    public func indices(where predicate: (NDArray)->Bool) -> [Int] {
        return enumerated().flatMap { i, array in
            predicate(array) ? i : nil
        }
    }
    
}
