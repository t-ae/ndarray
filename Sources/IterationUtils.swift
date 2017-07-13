
// MARK: - Unary
struct OffsetSequence: Sequence {
    typealias Iterator = OffsetIterator
    
    let shape: ArraySlice<Int>
    let strides: ArraySlice<Int>
    
    init(shape: ArraySlice<Int>, strides: ArraySlice<Int>) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == strides.count)
        assert(shape.startIndex == strides.startIndex && shape.endIndex == strides.endIndex)
        self.shape = shape
        self.strides = strides
    }
    
    init(shape: [Int], strides: [Int]) {
        assert(shape.count == strides.count)
        self.init(shape: shape[0..<shape.count],
                  strides: strides[0..<strides.count])
    }
    
    func makeIterator() -> OffsetIterator {
        return OffsetIterator(shape: shape, strides: strides)
    }
}

struct OffsetIterator: IteratorProtocol {
    typealias Element = Int
    
    let shape: ArraySlice<Int>
    let strides: ArraySlice<Int>
    var index: [Int]
    var offset: Int?
    
    init(shape: ArraySlice<Int>, strides: ArraySlice<Int>) {
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
    
    let shape: ArraySlice<Int>
    let lStrides: ArraySlice<Int>
    let rStrides: ArraySlice<Int>
    
    init(shape: ArraySlice<Int>, lStrides: ArraySlice<Int>, rStrides: ArraySlice<Int>) {
        assert(shape.all { $0 >= 0 })
        assert(shape.count == lStrides.count && shape.count == rStrides.count)
        assert(shape.startIndex == lStrides.startIndex && shape.endIndex == lStrides.endIndex)
        assert(shape.startIndex == rStrides.startIndex && shape.endIndex == rStrides.endIndex)
        self.shape = shape
        self.lStrides = lStrides
        self.rStrides = rStrides
    }
    
    init(shape: [Int], lStrides: [Int], rStrides: [Int]) {
        assert(shape.count == lStrides.count && shape.count == rStrides.count)
        self.init(shape: shape[0..<shape.count],
                  lStrides: lStrides[0..<lStrides.count],
                  rStrides: rStrides[0..<rStrides.count])
        
    }
    
    func makeIterator() -> BinaryOffsetIterator {
        return BinaryOffsetIterator(shape: shape, lStrides: lStrides, rStrides: rStrides)
    }
}


struct BinaryOffsetIterator: IteratorProtocol {
    typealias Element = (l: Int, r: Int)
    
    let shape: ArraySlice<Int>
    let lStrides: ArraySlice<Int>
    let rStrides: ArraySlice<Int>
    var index: [Int]
    var offset: (l: Int, r: Int)?
    
    init(shape: ArraySlice<Int>, lStrides: ArraySlice<Int>, rStrides: ArraySlice<Int>) {
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
            let si = i + shape.startIndex
            if index[i] < shape[si]-1 {
                index[i] += 1
                offset!.l += lStrides[si]
                offset!.r += rStrides[si]
                break
            } else if i > 0 {
                index[i] = 0
                offset!.l -= lStrides[si]*(shape[si]-1)
                offset!.r -= rStrides[si]*(shape[si]-1)
            } else {
                offset = nil
            }
        }
        
        return ret
    }
}
