import Accelerate

public func _matmul(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    
    precondition(lhs.ndim > 1 && rhs.ndim > 1)
    
    let M = Int32(lhs.shape[lhs.ndim-2])
    let N = Int32(rhs.shape[rhs.ndim-1])
    let K = Int32(lhs.shape[lhs.ndim-1])
    precondition(rhs.shape[rhs.ndim-2] == Int(K))
    
    let (lhs, rhs) = matmulBroadcast(lhs, rhs)
    let majorShape = [Int](lhs.shape.dropLast(2))
    
    let matrixSize = Int(M*N)
    let majorSize = majorShape.prod()
    
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: majorSize*matrixSize)
    defer { dst.deallocate(capacity: majorSize*matrixSize) }
    
    let lPtr = UnsafePointer(lhs.data)
    let rPtr = UnsafePointer(rhs.data)
    let lOffsets = getOffsets(shape: majorShape, strides: [Int](lhs.strides.dropLast(2)))
    let rOffsets = getOffsets(shape: majorShape, strides: [Int](rhs.strides.dropLast(2)))
    var dstPtr = dst
    for (lo, ro) in zip(lOffsets, rOffsets) {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1, lPtr.advanced(by: lo), K,
                    rPtr.advanced(by: ro), N,
                    0, dstPtr, N)
        dstPtr += matrixSize
    }
    
    return NDArray(shape: majorShape + [Int(M), Int(N)],
                   elements: [Float](UnsafeBufferPointer(start: dst, count: majorSize*matrixSize)))
}

infix operator <*>: MultiplicationPrecedence

/// Matlix multiplication
public func <*>(lhs: NDArray, rhs: NDArray) -> NDArray {
    return _matmul(lhs, rhs)
}

extension NDArray {
    /// Matlix multiplication
    func matmul(_ rhs: NDArray) -> NDArray {
        return _matmul(self, rhs)
    }
}

private func matmulBroadcast(_ lhs: NDArray, _ rhs: NDArray) -> (NDArray, NDArray) {
    
    let lElements = gatherElements(lhs)
    let rElements = gatherElements(rhs)
    
    var lShape = lhs.shape
    var rShape = rhs.shape
    var lStrides = continuousStrides(shape: lhs.shape)
    var rStrides = continuousStrides(shape: rhs.shape)
    
    let d = lShape.count - rShape.count
    if d < 0 {
        lShape = [Int](repeating: 1, count: -d) + lShape
        lStrides = [Int](repeating: 0, count: -d) + lStrides
    } else if d > 0 {
        rShape = [Int](repeating: 1, count: d) + rShape
        rStrides = [Int](repeating: 0, count: d) + rStrides
    }
    
    for i in (0..<lShape.count-2).reversed() {
        if lShape[i] == rShape[i] {
            continue
        } else if lShape[i] == 1 {
            lShape[i] = rShape[i]
            lStrides[i] = 0
        } else if rShape[i] == 1 {
            rShape[i] = lShape[i]
            rStrides[i] = 0
        } else {
            preconditionFailure()
        }
    }
    
    return (NDArray(shape: lhs.shape, strides: lStrides, baseOffset: 0, data: lElements),
            NDArray(shape: rhs.shape, strides: rStrides, baseOffset: 0, data: rElements))
    
}
