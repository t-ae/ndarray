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

public struct OneSidedRange: NDArrayIndexElementProtocol {
    var start: Int?
    var end: Int?
    
    init(start: Int?, end: Int?) {
        self.start = start
        self.end = end
    }
    
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(start: start, end: end)
    }
}

prefix operator ..<
public prefix func ..<(rhs: Int) -> OneSidedRange {
    return OneSidedRange(start: nil, end: rhs)
}

prefix operator ..<-
public prefix func ..<-(rhs: Int) -> OneSidedRange {
    return OneSidedRange(start: nil, end: -rhs)
}

postfix operator ...
public postfix func ...(lhs: Int) -> OneSidedRange {
    return OneSidedRange(start: lhs, end: nil)
}

extension Int: NDArrayIndexElementProtocol {
    public func getNDArrayIndexElement() -> NDArrayIndexElement {
        return NDArrayIndexElement(single: self)
    }
}
extension CountableRange: NDArrayIndexElementProtocol where Element == Int {
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

public func ~>(range: CountableRange<Int>, stride: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: range.startIndex, end: range.endIndex, stride: stride)
}

public func ~>(range: OneSidedRange, stride: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: range.start, end: range.end, stride: stride)
}

public prefix func ~>(stride: Int) -> NDArrayIndexElement {
    return NDArrayIndexElement(start: nil, end: nil, stride: stride)
}

public func ~>-(range: CountableRange<Int>, stride: Int) -> NDArrayIndexElement {
    return range ~> (-stride)
}

public func ~>-(range: OneSidedRange, stride: Int) -> NDArrayIndexElement {
    return range ~> (-stride)
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
