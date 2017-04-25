import Accelerate

/// Matrix multiplication
///
/// If either argument is N-D, N > 2, it is treated as a stack of matrices residing 
/// in the last two indexes and broadcast accordingly.
///
/// 1-D or 0-D arrays are not allowed.
public func matmul(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    
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
    
    let lPtr = UnsafePointer(lhs.data) + lhs.baseOffset
    let rPtr = UnsafePointer(rhs.data) + rhs.baseOffset
    let lOffsets = getOffsets(shape: majorShape, strides: [Int](lhs.strides.dropLast(2)))
    let rOffsets = getOffsets(shape: majorShape, strides: [Int](rhs.strides.dropLast(2)))
    var dstPtr = dst
    for (lo, ro) in zip(lOffsets, rOffsets) {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1, lPtr + lo, K,
                    rPtr + ro, N,
                    0, dstPtr, N)
        dstPtr += matrixSize
    }
    
    return NDArray(shape: majorShape + [Int(M), Int(N)],
                   elements: [Float](UnsafeBufferPointer(start: dst, count: majorSize*matrixSize)))
}

infix operator <*>: MultiplicationPrecedence

/// Matlix multiplication
///
/// See `matmul`
public func <*>(lhs: NDArray, rhs: NDArray) -> NDArray {
    return matmul(lhs, rhs)
}

private func matmulBroadcast(_ lhs: NDArray, _ rhs: NDArray) -> (NDArray, NDArray) {
    
    var lShape = lhs.shape
    var lStrides: [Int]
    let lBaseOffset: Int
    let lData: [Float]
    if lhs.strides == [lhs.shape[lhs.ndim-1], 1] {
        // submatrices are continuous
        lData = lhs.data
        lStrides = lhs.strides
        lBaseOffset = lhs.baseOffset
    } else {
        lData = gatherElements(lhs)
        lStrides = continuousStrides(shape: lShape)
        lBaseOffset = 0
    }
    
    var rShape = rhs.shape
    var rStrides: [Int]
    let rBaseOffset: Int
    let rData: [Float]
    if rhs.strides == [rhs.shape[rhs.ndim-1], 1] {
        // submatrices are continuous
        rData = rhs.data
        rStrides = rhs.strides
        rBaseOffset = rhs.baseOffset
    } else {
        rData = gatherElements(rhs)
        rStrides = continuousStrides(shape: rShape)
        rBaseOffset = 0
    }
    
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
    
    let lhs = NDArray(shape: lShape,
                      strides: lStrides,
                      baseOffset: lBaseOffset,
                      data: lData)
    let rhs = NDArray(shape: rShape,
                      strides: rStrides,
                      baseOffset: rBaseOffset,
                      data: rData)
    
    return (lhs, rhs)
    
}
