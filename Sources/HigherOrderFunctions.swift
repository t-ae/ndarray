
import Foundation

extension NDArray {
    
    /// Apply `transform` for each elements.
    public func mapElements(_ transform: (Float)->Float) -> NDArray {
        var elements = gatherElements(self)
        elements.withUnsafeMutableBufferPointer {
            var p = $0.baseAddress!
            for _ in 0..<$0.count {
                p.pointee = transform(p.pointee)
                p += 1
            }
        }
        return NDArray(shape: shape, elements: elements)
    }
}
