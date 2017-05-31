
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
    
    init(shape: [Int], strides: [Int]) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == strides.count)
        self.shape = shape
        self.strides = strides
        self.index = [Int](repeating: 0, count: shape.count)
        if shape.contains(0) {
            self.offset = nil
        } else {
            self.offset = 0
        }
    }
    
    mutating func next() -> Int? {
        guard let ret = offset else {
            return nil
        }
        
        guard !shape.isEmpty else {
            offset = nil
            return 0
        }
        
        for i in (0..<index.count).reversed() {
            if index[i] < shape[i]-1 {
                index[i] += 1
                offset! += strides[i]
                break
            } else if i > 0 {
                index[i] = 0
                offset! -= strides[i]*(shape[i]-1)
            } else {
                offset = nil
            }
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
    var offset: (l: Int, r: Int)?
    
    init(shape: [Int], lStrides: [Int], rStrides: [Int]) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == lStrides.count && shape.count == rStrides.count)
        self.shape = shape
        self.lStrides = lStrides
        self.rStrides = rStrides
        self.index = [Int](repeating: 0, count: shape.count)
        if shape.contains(0) {
            self.offset = nil
        } else {
            self.offset = (0, 0)
        }
    }

    mutating func next() -> (l: Int, r: Int)? {
        guard let ret = offset else {
            return nil
        }
        
        guard !shape.isEmpty else {
            offset = nil
            return ret
        }
        
        for i in (0..<index.count).reversed() {
            if index[i] < shape[i]-1 {
                index[i] += 1
                offset!.l += lStrides[i]
                offset!.r += rStrides[i]
                break
            } else if i > 0 {
                index[i] = 0
                offset!.l -= lStrides[i]*(shape[i]-1)
                offset!.r -= rStrides[i]*(shape[i]-1)
            } else {
                offset = nil
            }
        }
        
        return ret
    }
}
