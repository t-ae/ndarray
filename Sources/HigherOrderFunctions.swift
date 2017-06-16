
import Foundation

extension NDArray {
    
    /// Apply `transform` for each elements.
    public func mapElements(_ transform: (Float)->Float) -> NDArray {
        var elements = gatherElements(self)
        elements.withUnsafeMutablePointer {
            var p = $0
            for _ in 0..<elements.count {
                p.pointee = transform(p.pointee)
                p += 1
            }
        }
        return NDArray(shape: shape, elements: elements)
    }
}
