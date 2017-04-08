
struct NDIndexSequence: Sequence {
    typealias Iterator = NDIndexIterator
    
    let shape: [Int]
    
    func makeIterator() -> NDIndexIterator {
        return NDIndexIterator(shape: shape)
    }
}

public struct NDIndexIterator: IteratorProtocol {
    public typealias Element = [Int]
    
    // reversed
    let shape: [Int]
    var ndIndex: [Int]?
    
    var end = false
    
    init(shape: [Int]) {
        self.shape = shape.reversed()
        ndIndex = [Int](repeating: 0, count: shape.count)
    }
    
    public mutating func next() -> [Int]? {
        
        let ret = ndIndex
        if !(ndIndex?.isEmpty ?? true) {
            // not nil and not empty
            // increment
            ndIndex![0] += 1
            for i in 0..<ndIndex!.count {
                guard ndIndex![i] == shape[i] else {
                    break
                }
                if i == ndIndex!.count-1 {
                    ndIndex = nil
                } else {
                    ndIndex![i+1] += 1
                    ndIndex![i] = 0
                }
            }
        } else if ndIndex?.isEmpty ?? false {
            // ndIndex == []
            ndIndex = nil
        }
        
        return ret?.reversed()
    }
}
