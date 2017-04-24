
extension NDArray: Sequence {
    
    public typealias Iterator = AnyIterator<NDArray>
    
    public func makeIterator() -> AnyIterator<NDArray> {
        precondition(!self.isScalar)
        
        var i = 0
        return AnyIterator{ _ in
            guard i < self.shape[0] else {
                return nil
            }
            let subarray = self[i]
            i += 1
            return subarray
        }
    }
}
