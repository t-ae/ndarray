import Accelerate

/// Matrix multiplication
///
/// If either argument is N-D, N > 2, it is treated as a stack of matrices residing 
/// in the last two indexes and broadcast accordingly.
///
/// 1-D or 0-D arrays are not allowed.
public func matmul(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    
    precondition(lhs.ndim > 1, "`lhs` is not matrix: shape: \(lhs.shape)")
    precondition(rhs.ndim > 1, "`rhs` is not matrix: shape: \(rhs.shape)")
    
    let M = Int32(lhs.shape[lhs.ndim-2])
    let N = Int32(rhs.shape[rhs.ndim-1])
    let K = Int32(lhs.shape[lhs.ndim-1])
    precondition(rhs.shape[rhs.ndim-2] == Int(K), "Incompatible shapes: \(lhs.shape) and \(rhs.shape)")
    
    let (lhs, rhs) = matmulBroadcast(lhs, rhs)
    let majorShape = [Int](lhs.shape.dropLast(2))
    
    let matrixSize = Int(M*N)
    let majorSize = majorShape.prod()
    
    var dst = NDArrayData<Float>(size: majorSize*matrixSize)
    
    let lda = Int32(lhs.strides[lhs.ndim-2])
    let ldb = Int32(rhs.strides[rhs.ndim-2])
    
    let offsets = BinaryOffsetSequence(shape: majorShape,
                                       lStrides: [Int](lhs.strides.dropLast(2)),
                                       rStrides: [Int](rhs.strides.dropLast(2)))

    withUnsafePointers(lhs, rhs) { lp, rp in
        dst.withUnsafeMutablePointer {
            var dstPtr = $0
            for (lo, ro) in offsets {
                cblas_sgemm(CblasRowMajor,
                            CblasNoTrans, CblasNoTrans,
                            M, N, K,
                            1, lp + lo, lda,
                            rp + ro, ldb,
                            0, dstPtr, N)
                dstPtr += matrixSize
            }
        }
    }
    
    return NDArray(shape: majorShape + [Int(M), Int(N)], elements: dst)
}

infix operator |*|: MultiplicationPrecedence

/// Matlix multiplication
///
/// See `matmul`
public func |*|(lhs: NDArray, rhs: NDArray) -> NDArray {
    return matmul(lhs, rhs)
}

private func matmulBroadcast(_ lhs: NDArray, _ rhs: NDArray) -> (NDArray, NDArray) {
    
    var (lhs, rhs) = (lhs, rhs)
    
    // lda >= N
    if needsGather(shape: lhs.shape, strides: lhs.strides) {
        lhs.data = gatherElements(lhs)
        lhs.strides = getContiguousStrides(shape: lhs.shape)
        lhs.baseOffset = 0
    }
    if lhs.shape[lhs.ndim-2] == 1 { // set appropriate lda
        lhs.strides[lhs.ndim-2] = lhs.shape.last!
    }
    // ldb >= M
    if needsGather(shape: rhs.shape, strides: rhs.strides) {
        rhs.data = gatherElements(rhs)
        rhs.strides = getContiguousStrides(shape: rhs.shape)
        rhs.baseOffset = 0
    }
    if rhs.shape[rhs.ndim-2] == 1 { // set appropriate ldb
        rhs.strides[rhs.ndim-2] = rhs.shape.last!
    }
    
    let d = lhs.shape.count - rhs.shape.count
    if d < 0 {
        lhs.shape = [Int](repeating: 1, count: -d) + lhs.shape
        lhs.strides = [Int](repeating: 0, count: -d) + lhs.strides
    } else if d > 0 {
        rhs.shape = [Int](repeating: 1, count: d) + rhs.shape
        rhs.strides = [Int](repeating: 0, count: d) + rhs.strides
    }
    
    for i in (0..<lhs.shape.count-2).reversed() {
        if lhs.shape[i] == rhs.shape[i] {
            continue
        } else if lhs.shape[i] == 1 {
            lhs.shape[i] = rhs.shape[i]
            lhs.strides[i] = 0
        } else if rhs.shape[i] == 1 {
            rhs.shape[i] = lhs.shape[i]
            rhs.strides[i] = 0
        } else {
            preconditionFailure()
        }
    }
    
    return (lhs, rhs)
}

private func needsGather(shape: [Int], strides: [Int]) -> Bool {
    assert(shape.count == strides.count)
    assert(shape.count > 1)
    let ndim = shape.count
    
    guard shape[ndim-1] == 1 || strides[ndim-1] == 1 else {
        return true
    }
    
    guard shape[ndim-2] == 1 || strides[ndim-2] >= shape[ndim-1] else {
        return true
    }
    
    return false
}
