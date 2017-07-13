
struct NDArrayData<T>: Collection {
    var buffer: ManagedBuffer<Int, T>
    
    init(size: Int) {
        buffer = ManagedBuffer.create(minimumCapacity: size) { _ in size }
    }
    
    init (_ array: [T]) {
        self.init(size: array.count)
        buffer.withUnsafeMutablePointerToElements { buf in
            buf.initialize(from: array, count: array.count)
        }
    }
    
    init(value: T, count: Int) {
        self.init(size: count)
        buffer.withUnsafeMutablePointerToElements { buf in
            buf.initialize(to: value, count: count)
        }
    }
    
    mutating func ensureUniquelyReferenced() {
        guard !isKnownUniquelyReferenced(&buffer) else {
            return
        }
        buffer = ManagedBuffer.create(minimumCapacity: count) { buf in
            withUnsafePointer { pointer in
                buf.withUnsafeMutablePointerToElements { dst in
                    dst.initialize(from: pointer, count: count)
                }
            }
            return count
        }
    }
    
    var count: Int {
        return buffer.header
    }
    
    var startIndex: Int { return 0 }
    var endIndex: Int { return count }
    
    func index(after index: Int) -> Int {
        return index + 1
    }
    
    subscript(index: Int) -> T {
        get {
            return buffer.withUnsafeMutablePointerToElements { $0[index] }
        }
        set {
            ensureUniquelyReferenced()
            buffer.withUnsafeMutablePointerToElements { $0[index] = newValue }
        }
    }
    
    subscript(range: CountableRange<Int>) -> NDArrayData<T> {
        assert(range.startIndex >= 0 && range.endIndex <= count)
        let new = NDArrayData(size: range.count)
        withUnsafePointer { pointer in
            let src = pointer + range.startIndex
            new.buffer.withUnsafeMutablePointerToElements { buf in
                buf.initialize(from: src, count: range.count)
            }
        }
        
        return new
    }
    
    @inline(__always)
    func withUnsafePointer<R>(_ body: (UnsafePointer<T>) throws -> R) rethrows -> R {
        return try buffer.withUnsafeMutablePointerToElements {
            try body(UnsafePointer($0))
        }
    }
    
    @inline(__always)
    mutating func withUnsafeMutablePointer<R>(_ body: (UnsafeMutablePointer<T>) throws -> R) rethrows -> R {
        ensureUniquelyReferenced()
        return try buffer.withUnsafeMutablePointerToElements(body)
    }
    
    func asArray() -> [T] {
        return withUnsafePointer { [T](UnsafeBufferPointer(start: $0, count: count)) }
    }
}
