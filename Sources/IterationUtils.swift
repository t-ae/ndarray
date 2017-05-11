
// MARK: Index
struct NDIndexSequence: Sequence {
    typealias Iterator = NDINdexIterator
    
    let shape: [Int]
    
    init(shape: [Int]) {
        assert(shape.all { $0 >= 0 })
        self.shape = shape
    }
    
    func makeIterator() -> NDINdexIterator {
        return NDINdexIterator(shape: shape)
    }
}

struct NDINdexIterator: IteratorProtocol {
    typealias Element = [Int]
    
    let shape: [Int]
    let last: Int
    var index: [Int]?
    
    init(shape: [Int]) {
        assert(shape.all { $0 >= 0 })
        self.shape = shape
        self.last = shape.count - 1
        if shape.contains(0) {
            self.index = nil
        } else {
            self.index = [Int](repeating: 0, count: shape.count)
        }
    }
    
    mutating func next() -> [Int]? {
        
        guard let ret = index else {
            return nil
        }
        guard !shape.isEmpty else {
            index = nil
            return []
        }
        
        index![last] += 1
        for i in 0..<last {
            guard index![last-i] >= shape[last-i] else {
                break
            }
            index![last-i] = 0
            index![last-i-1] += 1
        }
        if index![0] == shape[0] {
            index = nil
        }
        
        return ret
    }
}


// MARK: Offset
struct OffsetSequence: Sequence {
    typealias Iterator = OffsetIterator
    let shape: [Int]
    let strides: [Int]
    init(shape: [Int], strides: [Int]) {
        self.shape = shape
        self.strides = strides
    }
    
    func makeIterator() -> OffsetIterator {
        return OffsetIterator(shape: shape, strides: strides)
    }
}

struct OffsetIterator: IteratorProtocol {
    typealias Element = Int
    
    let shape: [Int]
    let strides: [Int]
    var index: [Int]
    var offset: Int?
    let last: Int
    
    init(shape: [Int], strides: [Int]) {
        self.shape = shape
        self.strides = strides
        self.index = [Int](repeating: 0, count: shape.count)
        self.offset = 0
        self.last = index.count - 1
    }
    
    mutating func next() -> Int? {
        guard let ret = offset else {
            return nil
        }
        
        guard !shape.isEmpty else {
            offset = nil
            return 0
        }
        
        index[last] += 1
        offset! += strides[last]
        for i in 0..<last {
            guard index[last-i] >= shape[last-i] else {
                break
            }
            index[last-i] = 0
            offset! -= strides[last-i]*shape[last-i]
            index[last-i-1] += 1
            offset! += strides[last-i-1]
        }
        
        if index[0] == shape[0] {
            offset = nil
        }
        
        return ret
    }
}
