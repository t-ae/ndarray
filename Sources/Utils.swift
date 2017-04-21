
import Accelerate

/// check if elements are aligned continuously
func isContinuous(shape: [Int], strides: [Int]) -> Bool {
    return shape.isEmpty || (strides.last == 1 && isStrided(shape: shape, strides: strides))
}

/// check if whole elements are strided
func isStrided(shape: [Int], strides: [Int]) -> Bool {
    return shape.count == stridedDims(shape: shape, strides: strides)
}

// check if elements are densely placed
func isDense(shape: [Int], strides: [Int]) -> Bool {
    return Set(strides) == Set(continuousStrides(shape: shape))
}

/// Get continuous strides
func continuousStrides(shape: [Int]) -> [Int] {
    guard !shape.isEmpty else {
        return []
    }
    var strides = [1]
    for s in shape.dropFirst().reversed() {
        strides.insert(strides[0]*s, at: 0)
    }
    return strides
}

/// Get offset
func indexOffset(strides: [Int], ndIndex: [Int]) -> Int {
    precondition(strides.count == ndIndex.count)
    return zip(ndIndex, strides)
        .map(*)
        .reduce(0, +)
}

/// get indices in row major order
func getIndices(shape: [Int]) -> [[Int]] {
    guard !shape.isEmpty else {
        return [[]]
    }
    guard !shape.contains(0) else {
        return []
    }
    
    var index = [Int](repeating: 0, count: shape.count)
    var indices: [[Int]] = []
    let last = index.count - 1
    
    while true {
        indices.append(index)
        index[last] += 1
        for i in 0..<last {
            guard index[last-i] >= shape[last-i] else {
                break
            }
            index[last-i] = 0
            index[last-i-1] += 1
        }
        if index[0] == shape[0] {
            break
        }
    }
    
    return indices
}

// get offsets in row major order
func getOffsets(shape: [Int], strides: [Int]) -> [Int] {
    precondition(shape.count == strides.count)
    guard !shape.isEmpty else {
        return [0]
    }
    guard !shape.contains(0) else {
        return []
    }
    
    var index = [Int](repeating: 0, count: shape.count)
    var offset = 0
    var offsets = [Int]()
    let last = index.count - 1
    
    while true {
        offsets.append(offset)
        index[last] += 1
        offset += strides[last]
        
        for i in 0..<last {
            guard index[last-i] >= shape[last-i] else {
                break
            }
            index[last-i] = 0
            offset -= strides[last-i]*shape[last-i]
            index[last-i-1] += 1
            offset += strides[last-i-1]
        }
        if index[0] == shape[0] {
            break
        }
    }
    
    return offsets
}

/// Calculate how many dims are strided
func stridedDims(shape: [Int], strides: [Int]) -> Int {
    precondition(shape.count == strides.count)
    var stridedDims = 0
    guard var stride = strides.last else {
        return 0
    }
    for (s, str) in zip(shape.reversed(), strides.reversed()) {
        if s == 1 {
            stridedDims += 1
        }else if stride == str {
            stridedDims += 1
            stride *= s
        } else {
            break
        }
    }
    return stridedDims
}

/// Gather elements
func gatherElements(_ arg: NDArray, forceUniqueReference: Bool = false) -> [Float] {
    
    let volume = arg.volume
    
    if isContinuous(shape: arg.shape, strides: arg.strides) {
        if arg.baseOffset == 0 && volume == arg.data.count {
            if forceUniqueReference {
                let dst = UnsafeMutablePointer<Float>.allocate(capacity: arg.data.count)
                defer { dst.deallocate(capacity: arg.data.count) }
                memcpy(dst, arg.data, arg.data.count*MemoryLayout<Float>.size)
                return Array(UnsafeBufferPointer(start: dst, count: arg.data.count))
            } else {
                return arg.data
            }
        } else {
            let start = arg.baseOffset
            let end = start + volume
            return Array(arg.data[start..<end])
        }
    } else {
        // Separate scattered major shape and strided minor shape
        let minorDims = stridedDims(shape: arg.shape, strides: arg.strides)
        let majorShape = [Int](arg.shape.dropLast(minorDims))
        let majorStrides = [Int](arg.strides.dropLast(minorDims))
        let minorZeros = [Int](repeating: 0, count: minorDims)
        
        let stride = Int32(arg.strides.last!)
        let count = arg.shape.suffix(minorDims).reduce(1, *)
        
        let dst = UnsafeMutablePointer<Float>.allocate(capacity: volume)
        defer { dst.deallocate(capacity: volume) }
        
        let offsets = getOffsets(shape: majorShape, strides: majorStrides)
        let numProc = ProcessInfo.processInfo.activeProcessorCount
        let blockSize = Int(ceil(Float(offsets.count) / Float(numProc)))
        
        DispatchQueue.concurrentPerform(iterations: numProc) { i in
            var dstPtr = dst.advanced(by: i*blockSize*count)
            let end = i*blockSize + min(blockSize, offsets.count - i*blockSize)
            
            guard i*blockSize < end else { // can be empty
                return
            }
            for oi in i*blockSize..<end {
                let offset = offsets[oi] + arg.baseOffset
                let src = UnsafePointer(arg.data).advanced(by: offset)
                
                cblas_scopy(Int32(count), src, stride, dstPtr, 1)
                dstPtr += count
            }
        }
        
        return [Float](UnsafeBufferPointer(start: dst, count: volume))
    }
}

/// Return normalized index
/// - Check all numbers in valid range
/// - Process minus number
func normalizeIndex(shape: [Int], ndIndex: [Int]) -> [Int] {
    precondition(shape.count == ndIndex.count)
    
    var ndIndex = ndIndex
    for i in 0..<ndIndex.count {
        if ndIndex[i] < -shape[i] || ndIndex[i] >= shape[i] {
            preconditionFailure()
        }
        if ndIndex[i] < 0 {
            ndIndex[i] += shape[i]
        }
    }
    return ndIndex
}

extension Array where Element == Bool {
    func all() -> Bool {
        for e in self {
            if !e {
                return false
            }
        }
        return true
    }
    
    func some() -> Bool {
        for e in self {
            if e {
                return true
            }
        }
        return false
    }
}

extension Array {
    func removed(at index: Int) -> Array {
        var ret = self
        ret.remove(at: index)
        return ret
    }
}
