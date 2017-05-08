
extension NDArray {
    
    public func select(_ indices: [Int]) -> NDArray {
        return NDArray.stack(indices.map{ self[$0] })
    }
    
    public func select(where predicate: (NDArray)->Bool) -> NDArray {
        return NDArray.stack(self.filter(predicate))
    }
    
}
