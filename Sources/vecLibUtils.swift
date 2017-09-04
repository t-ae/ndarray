
import Accelerate

typealias vvUnaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

func apply(_ arg: NDArray, _ vvfunc: vvUnaryFunc) -> NDArray {
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        let count = denseDataCount(shape: arg.shape, strides: arg.strides)
        var dst = [Float](repeating: 0, count: count)
        
        var _count = Int32(count)
        arg.withUnsafePointer { src in
            dst.withUnsafeMutableBufferPointer {
                vvfunc($0.baseAddress!, src, &_count)
            }
        }
        
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: dst)
    } else if arg.strides.last == 1 {
        let volume = arg.volume
        let strDims = getStridedDims(shape: arg.shape, strides: arg.strides)
        
        let majorShape = arg.shape.dropLast(strDims)
        let majorStrides = arg.strides.dropLast(strDims)
        let blockSize = arg.shape.suffix(strDims).prod()
        
        var dst = [Float](repeating: 0, count: volume)
        
        let offsets = OffsetSequence(shape: majorShape, strides: majorStrides)
        var _blockSize = Int32(blockSize)
        
        arg.withUnsafePointer { src in
            dst.withUnsafeMutableBufferPointer {
                var dst = $0.baseAddress!
                for offset in offsets {
                    let src = src + offset
                    
                    vvfunc(dst, src, &_blockSize)
                    dst += blockSize
                }
            }
        }
        
        return NDArray(shape: arg.shape, elements: dst)
        
    } else {
        let volume = arg.volume
        var count = Int32(volume)
        var elements = gatherElements(arg)
        
        elements.withUnsafeMutableBufferPointer {
            vvfunc($0.baseAddress!, $0.baseAddress!, &count)
        }
        
        return NDArray(shape: arg.shape, elements: elements)
    }
}

typealias vvBinaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

func apply(_ lhs: NDArray, _ rhs: NDArray, _ vvfunc: vvBinaryFunc) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    
    if lhs.strides.last == 1 && rhs.strides.last == 1 {
        let strDims = min(getStridedDims(shape: lhs.shape, strides: lhs.strides),
                          getStridedDims(shape: rhs.shape, strides: rhs.strides))
        
        let majorShape = lhs.shape.dropLast(strDims)
        let lMajorStrides = lhs.strides.dropLast(strDims)
        let rMajorStrides = rhs.strides.dropLast(strDims)
        let blockSize = lhs.shape.suffix(strDims).prod()
        
        let offsets = BinaryOffsetSequence(shape: majorShape, lStrides: lMajorStrides, rStrides: rMajorStrides)
        var _blockSize = Int32(blockSize)
        
        var dst = [Float](repeating: 0, count: volume)
        
        lhs.withUnsafePointer { lp in
            rhs.withUnsafePointer { rp in
                dst.withUnsafeMutableBufferPointer {
                    var dst = $0.baseAddress!
                    for (lo, ro) in offsets {
                        vvfunc(dst, lp + lo, rp + ro, &_blockSize)
                        dst += blockSize
                    }
                }
            }
        }
        
        return NDArray(shape: lhs.shape, elements: dst)
    } else {
        var _volume = Int32(volume)
        
        var lElements = gatherElements(lhs)
        let rElements = gatherElements(rhs)
        
        lElements.withUnsafeMutableBufferPointer {
            let lp = $0.baseAddress!
            rElements.withUnsafeBufferPointer {
                let rp = $0.baseAddress!
                vvfunc(lp, lp, rp, &_volume)
            }
        }
        
        return NDArray(shape: lhs.shape, elements: lElements)
    }
}
