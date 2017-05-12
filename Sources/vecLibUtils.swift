
import Accelerate

typealias vvUnaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

func apply(_ arg: NDArray, _ vvfunc: vvUnaryFunc) -> NDArray {
    
    if isDense(shape: arg.shape, strides: arg.strides) {
        var count = Int32(arg.data.count)
        let src = arg.startPointer
        
        var dst = NDArrayData(size: arg.data.count)
        
        dst.withUnsafeMutablePointer {
            vvfunc($0, src, &count)
        }
        
        return NDArray(shape: arg.shape,
                       strides: arg.strides,
                       baseOffset: 0,
                       data: dst)
    } else if arg.strides.last == 1 {
        let volume = arg.volume
        let strDims = getStridedDims(shape: arg.shape, strides: arg.strides)
        
        let majorShape = [Int](arg.shape.dropLast(strDims))
        let majorStrides = [Int](arg.strides.dropLast(strDims))
        let blockSize = arg.shape.suffix(strDims).prod()
        
        var dst = NDArrayData(size: volume)
        
        let offsets = OffsetSequence(shape: majorShape, strides: majorStrides)
        var _blockSize = Int32(blockSize)
        
        let src = arg.startPointer
        dst.withUnsafeMutablePointer { dst in
            var dst = dst
            for offset in offsets {
                let src = src + offset
                
                vvfunc(dst, src, &_blockSize)
                dst += blockSize
            }
        }
        
        return NDArray(shape: arg.shape, elements: dst)
        
    } else {
        let volume = arg.volume
        var count = Int32(volume)
        let elements = gatherElements(arg)
        
        var dst = NDArrayData(size: volume)
        
        dst.withUnsafeMutablePointer {
            vvfunc($0, elements.pointer, &count)
        }
        
        return NDArray(shape: arg.shape, elements: dst)
    }
}

typealias vvBinaryFunc = (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Float>, UnsafePointer<Int32>) -> Void

func apply(_ lhs: NDArray, _ rhs: NDArray, _ vvfunc: vvBinaryFunc) -> NDArray {
    let (lhs, rhs) = broadcast(lhs, rhs)
    
    let volume = lhs.volume
    var dst = NDArrayData(size: volume)
    
    if lhs.strides.last == 1 && rhs.strides.last == 1 {
        let strDims = min(getStridedDims(shape: lhs.shape, strides: lhs.strides),
                          getStridedDims(shape: rhs.shape, strides: rhs.strides))
        
        let majorShape = [Int](lhs.shape.dropLast(strDims))
        let lMajorStrides = [Int](lhs.strides.dropLast(strDims))
        let rMajorStrides = [Int](rhs.strides.dropLast(strDims))
        let blockSize = lhs.shape.suffix(strDims).prod()
        
        let lOffsets = OffsetSequence(shape: majorShape, strides: lMajorStrides)
        let rOffsets = OffsetSequence(shape: majorShape, strides: rMajorStrides)
        var _blockSize = Int32(blockSize)
        
        let lSrc = lhs.startPointer
        let rSrc = rhs.startPointer
        dst.withUnsafeMutablePointer { dst in
            var dst = dst
            for (lo, ro) in zip(lOffsets, rOffsets) {
                vvfunc(dst, lSrc + lo, rSrc + ro, &_blockSize)
                dst += blockSize
            }
        }
        
        return NDArray(shape: lhs.shape, elements: dst)
    } else {
        var _volume = Int32(volume)
        
        let lElements = gatherElements(lhs)
        let rElements = gatherElements(rhs)
        
        dst.withUnsafeMutablePointer {
            vvfunc($0, lElements.pointer, rElements.pointer, &_volume)
        }
        
        return NDArray(shape: lhs.shape, elements: dst)
    }
}
