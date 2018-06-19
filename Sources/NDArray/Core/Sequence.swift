extension NDArray: Sequence {
    
    public typealias Iterator = NDArrayIterator
    
    public func makeIterator() -> NDArrayIterator {
        return NDArrayIterator(array: self)
    }
}

public struct NDArrayIterator: IteratorProtocol {
    public typealias Element = NDArray
    
    let array: NDArray
    var index: Int
    
    init(array: NDArray) {
        precondition(!array.isScalar, "Can't iterate scalar.")
        self.array = array
        self.index = 0
    }
    
    public mutating func next() -> NDArray? {
        guard index < array.shape[0] else {
            return nil
        }
        let subarray = array[index]
        index += 1
        return subarray
    }
}
