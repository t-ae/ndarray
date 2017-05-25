
// MARK: - Offset
struct OffsetSequence: Sequence {
    typealias Iterator = OffsetIterator
    let shape: [Int]
    let strides: [Int]
    init(shape: [Int], strides: [Int]) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == strides.count)
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
        assert(shape.all { $0 >= 0 })
        assert(shape.count == strides.count)
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
            index[last-i-1] += 1
            offset! += strides[last-i-1] - strides[last-i]*shape[last-i]
        }
        
        if index[0] == shape[0] {
            offset = nil
        }
        
        return ret
    }
}

struct BinaryOffsetSequence: Sequence {
    typealias Iterator = BinaryOffsetIterator
    
    let shape: [Int]
    let lStrides: [Int]
    let rStrides: [Int]
    
    init(shape: [Int], lStrides: [Int], rStrides: [Int]) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == lStrides.count && shape.count == rStrides.count)
        self.shape = shape
        self.lStrides = lStrides
        self.rStrides = rStrides
    }
    
    func makeIterator() -> BinaryOffsetIterator {
        return BinaryOffsetIterator(shape: shape, lStrides: lStrides, rStrides: rStrides)
    }
}


struct BinaryOffsetIterator: IteratorProtocol {
    typealias Element = (l: Int, r: Int)
    
    let shape: [Int]
    let lStrides: [Int]
    let rStrides: [Int]
    var index: [Int]
    let last: Int
    var offset: (l: Int, r: Int)?
    
    init(shape: [Int], lStrides: [Int], rStrides: [Int]) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == lStrides.count && shape.count == rStrides.count)
        self.shape = shape
        self.lStrides = lStrides
        self.rStrides = rStrides
        self.index = [Int](repeating: 0, count: shape.count)
        self.last = index.count - 1
        self.offset = (0, 0)
    }

    mutating func next() -> (l: Int, r: Int)? {
        guard let ret = offset else {
            return nil
        }
        
        guard !shape.isEmpty else {
            offset = nil
            return ret
        }
        
        index[last] += 1
        offset = (l: offset!.l + lStrides[last],
                  r: offset!.r + rStrides[last])
        for i in 0..<last {
            guard index[last-i] >= shape[last-i] else {
                break
            }
            index[last-i] = 0
            index[last-i-1] += 1
            offset = (l: offset!.l + lStrides[last-i-1] - lStrides[last-i]*shape[last-i],
                      r: offset!.r + rStrides[last-i-1] - rStrides[last-i]*shape[last-i])
        }
        
        if index[0] == shape[0] {
            offset = nil
        }
        
        return ret
    }
}
