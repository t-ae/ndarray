
import XCTest
@testable import NDArray

class PerformanceTests: XCTestCase {
    
    func testAsContiguousArrayPerformance() {
        
        // a = np.arange(10**6).reshape([10]*6).transpose()
        // timeit np.ascontiguousarray(a)
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()
        measure {
            _ = a.asContiguousArray()
        }
    }
}

// MARK: - Arithmetic
extension PerformanceTests {
    func testAddPerformance1() {
        
        // a = np.arange(10**6).reshape([10]*6).astype(float)
        // timeit a+a
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = a + a
        }
    }
    
    func testAddPerformance2() {
        
        // a = np.arange(10**6).reshape([10]*6).astype(float)
        // b = a.transpose()
        // timeit a+b
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        measure {
            _ = a + b
        }
    }
    
    func testAddPerformance3() {
        
        // a = np.arange(10**6).reshape([10]*6).astype(float)
        // b = a.transpose()
        // c = a.transpose([1,2,3,4,5,0])
        // timeit b+c
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        let c = a.transposed([1, 2, 3, 4, 5, 0])
        measure {
            _ = b + c
        }
    }
    
    func testNegPerformance1() {
        
        // a = np.arange(10**6).reshape([10]*6).astype(float)
        // timeit (-a)
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = -a
        }
    }
    
    func testNegPerformance2() {
        
        // a = np.arange(10**7).reshape([10]*7).astype(float).transpose()[1]
        // timeit (-a)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()[1]
        measure {
            _ = -a
        }
    }
}

// MARK: - FloatingPointFunctions
extension PerformanceTests {
    func testSqrtPerformance1() {
        
        // a = np.arange(10**6).reshape([10]*6)
        // timeit np.sqrt(a)
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = sqrt(a)
        }
    }
    
    func testSqrtPerformance2() {
        
        // a = np.arange(10**7).reshape([10]*7).transpose()[1]
        // timeit np.sqrt(a)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()[1]
        measure {
            _ = sqrt(a)
        }
    }
}

// MARK: - Reduce
extension PerformanceTests {
    func testMeanPerformance() {
        
        // a = np.arange(10**7).reshape([10]*7).astype(np.float32)
        // timeit np.mean(a)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = mean(a, along: 3)
        }
    }
}

// MARK: - LinearAlgebra
extension PerformanceTests {
    func testInvPerformance1() {
        
        // a = np.arange(10**5*2*2).reshape([10]*5+[2,2]).astype(np.float32)
        // timeit np.linalg.inv(a)
        
        let shape = [10, 10, 10, 10, 10, 2, 2]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = try! inv(a)
        }
    }
    
    func testInvPerformance2() {
        
        // a = np.arange(10**5*2*2).reshape([10]*5+[2,2]).transpose([0, 1, 2, 3, 4, 6, 5]).astype(np.float32)
        // timeit np.linalg.inv(a)
        
        let shape = [10, 10, 10, 10, 10, 2, 2]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed([0, 1, 2, 3, 4, 6, 5])
        measure {
            _ = try! inv(a)
        }
    }
}

// MARK: - Stack
extension PerformanceTests {
    func testStackPerformance() {
        
        // a = np.arange(10**5).reshape([10]*5)
        // timeit np.stack([a, a, a, a], -1)
        
        let shape = [10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = NDArray.stack([a, a, a, a, a], newAxis: -1)
        }
    }
}

// Clip
extension PerformanceTests {
    func testsClippedPerformance() {
        
        // a = np.arange(10**7).reshape([10]*7).astype(np.float32)
        // timeit np.clip(a, 100, 1000)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()
        measure {
            _ = a.clipped(low: 100, high: 1000)
        }
    }
    
    func testsMaximumPerformance() {
        
        // a = np.arange(10**6).reshape([10]*6).astype(np.float32)
        // b = a.transpose()
        // timeit np.maximum(a, b)
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        measure {
            _ = maximum(a, b)
        }
    }
}
