
import Accelerate

/// Get minimums for each pair elements
public func minimum(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_vmin)
}

/// Get maximums for each pair elements
public func maximum(_ lhs: NDArray, _ rhs: NDArray) -> NDArray {
    return apply(lhs, rhs, vDSP_vmax)
}

extension NDArray {
    /// Clip lower values.
    public func clipped(low: Float) -> NDArray {
        return clip(self, low: low, high: Float.infinity)
    }
    
    /// Clip higher values.
    public func clipped(high: Float) -> NDArray {
        return clip(self, low: -Float.infinity, high: high)
    }
    
    /// Clip lower and higher values.
    public func clipped(low: Float, high: Float) -> NDArray {
        return clip(self, low: low, high: high)
    }
}

private func clip(_ array: NDArray, low: Float, high: Float) -> NDArray {
    
    var low = low
    var high = high
    
    let f: vDSP_unary_func = { sp, ss, dp, ds, len in
        vDSP_vclip(sp, ss, &low, &high, dp, ds, len)
    }
    
    return apply(array, f)
}
