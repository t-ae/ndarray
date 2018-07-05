public protocol NDArrayIndexElementProtocol {
    func getNDArrayIndexElement() -> NDArrayIndexElement
}

public struct NDArrayIndexElement: NDArrayIndexElementProtocol {
    var start: Int?
    var end: Int?
    var stride: Int?
    
    // strided range index
    init(start: Int?, end: Int?, stride: Int = 1) {
        precondition(stride != 0)
        if let start = start, let end = end {
            precondition((end < 0 && start >= 0) || start <= end, "Invalid range: \(start)..<\(end)")
        }
        self.start = start
        self.end = end
        self.stride = stride
    }
    
    // Single index
    init(single: Int) {
        self.start = single
        self.end = nil
        self.stride = nil
    }
    
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return self
    }
}

extension Int: NDArrayIndexElementProtocol {
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(single: self)
    }
}

public protocol RangeIndexElement: NDArrayIndexElementProtocol {}

extension PartialRangeFrom: NDArrayIndexElementProtocol, RangeIndexElement where Bound == Int {
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(start: self.lowerBound, end: nil)
    }
}

extension PartialRangeUpTo: NDArrayIndexElementProtocol, RangeIndexElement where Bound == Int {
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(start: nil, end: upperBound)
    }
}

extension PartialRangeThrough: NDArrayIndexElementProtocol, RangeIndexElement where Bound == Int {
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(start: nil, end: upperBound+1)
    }
}

extension Range: NDArrayIndexElementProtocol, RangeIndexElement where Bound == Int {
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(start: startIndex, end: endIndex)
    }
}

precedencegroup StridePrecedence {
    associativity: left
    lowerThan: RangeFormationPrecedence
}

infix operator ~> : StridePrecedence
prefix operator ~>
infix operator ~>- : StridePrecedence
prefix operator ~>-

public func ~><I: RangeIndexElement>(range: I, stride: Int) -> NDArrayIndexElement {
    var index = range.getNDArrayIndexElement()
    index.stride = stride
    return index
}

public func ~>-<I: RangeIndexElement>(range: I, stride: Int) -> NDArrayIndexElement {
    return range ~> (-stride)
}

public prefix func ~>(stride: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: nil, end: nil, stride: stride)
}

public prefix func ~>-(stride: Int) -> NDArrayIndexElement {
    return ~>(-stride)
}

infix operator ...~>
infix operator ...~>-
public func ...~>(lhs: Int, rhs: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: lhs, end: nil, stride: rhs)
}
public func ...~>-(lhs: Int, rhs: Int) -> NDArrayIndexElement {
    return lhs ...~> -rhs
}
