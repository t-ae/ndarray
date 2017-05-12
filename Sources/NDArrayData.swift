
struct NDArrayData: Collection {
    var buffer: ManagedBuffer<Int, Float>
    
    init(size: Int) {
        buffer = ManagedBuffer.create(minimumCapacity: size) { _ in size }
    }
    
    init (_ array: [Float]) {
        self.init(size: array.count)
        buffer.withUnsafeMutablePointerToElements { buf in
            buf.initialize(from: array, count: array.count)
        }
    }
    
    init(value: Float, count: Int) {
        self.init(size: count)
        buffer.withUnsafeMutablePointerToElements { buf in
            buf.initialize(to: value, count: count)
        }
    }
    
    mutating func ensureUniquelyReferenced() {
        guard !isKnownUniquelyReferenced(&buffer) else {
            return
        }
        let count = self.count
        buffer = buffer.withUnsafeMutablePointerToElements { src in
            return ManagedBuffer.create(minimumCapacity: count) { buf in
                buf.withUnsafeMutablePointerToElements { dst in
                    dst.initialize(from: src, count: count)
                }
                return count
            }
        }
    }
    
    var count: Int {
        return buffer.withUnsafeMutablePointerToHeader { $0.pointee }
    }
    
    var startIndex: Int { return 0 }
    var endIndex: Int { return count }
    
    func index(after index: Int) -> Int {
        return index + 1
    }
    
    subscript(index: Int) -> Float {
        get {
            return buffer.withUnsafeMutablePointerToElements { $0[index] }
        }
        set {
            ensureUniquelyReferenced()
            buffer.withUnsafeMutablePointerToElements { $0[index] = newValue }
        }
    }
    
    subscript(range: CountableRange<Int>) -> NDArrayData {
        assert(range.startIndex >= 0 && range.endIndex < count)
        let new = NDArrayData(size: range.count)
        let src = pointer + range.startIndex
        new.buffer.withUnsafeMutablePointerToElements { buf in
            buf.initialize(from: src, count: range.count)
        }
        return new
    }
    
    mutating func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<Float>) throws -> R) rethrows -> R {
        ensureUniquelyReferenced()
        return try buffer.withUnsafeMutablePointerToElements(body)
    }
    
    var pointer: UnsafePointer<Float> {
        return buffer.withUnsafeMutablePointerToElements { UnsafePointer($0) }
    }
    
    func asArray() -> [Float] {
        return [Float](UnsafeBufferPointer(start: pointer, count: count))
    }
}
