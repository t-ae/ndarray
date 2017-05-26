
import XCTest
import Accelerate
@testable import NDArray

#if !SWIFT_PACKAGE
class PerformanceTests: XCTestCase {
    
    func testAsContiguousArray() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6).transpose()
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
    func testAdd1() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // timeit a+a
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = a + a
        }
    }
    
    func testAdd2() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // b = a.transpose()
        // timeit a+b
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        measure {
            _ = a + b
        }
    }
    
    func testAdd3() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
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
    
    func testAdd4() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // b = a.transpose()
        // c = a.transpose([1,2,3,4,5,0])
        // timeit b+c
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = a + 1
        }
    }
    
    func testNeg1() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // timeit (-a)
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = -a
        }
    }
    
    func testNeg2() {
        
        // a = np.arange(10**7, dtype=np.float32).reshape([10]*7).transpose()[1]
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
    func testSqrt1() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // timeit np.sqrt(a)
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = sqrt(a)
        }
    }
    
    func testSqrt2() {
        
        // a = np.arange(10**7, dtype=np.float32).reshape([10]*7).transpose()[1]
        // timeit np.sqrt(a)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()[1]
        measure {
            _ = sqrt(a)
        }
    }
}
    
// MARK - Pow
extension PerformanceTests {
    func testPow1() {
        
        // a = np.ones([128, 128, 128])
        // timeit a ** 2

        let a = NDArray.ones([128, 128, 128])
        measure {
            _ = a ** 2
        }
    }
    
    func testPow2() {
        
        // a = np.ones([128, 128, 128])
        // timeit 2 ** a
        
        let a = NDArray.ones([128, 128, 128])
        measure {
            _ = 2 ** a
        }
    }
    
    func testPow3() {
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // b = a.transpose()
        // timeit a**a
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = a ** a
        }
    }
    
    func testPow4() {
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
        // b = a.transpose()
        // timeit a+b
        
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        measure {
            _ = a ** b
        }
    }
}

// MARK: - Reduce
extension PerformanceTests {
    func testMean() {
        
        // a = np.arange(10**7, dtype=np.float32).reshape([10]*7)
        // timeit np.mean(a, 3)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = mean(a, along: 3)
        }
    }
}

// MARK: - LinearAlgebra
extension PerformanceTests {
    func testInv1() {
        
        // a = np.arange(10**5*2*2, dtype=np.float32).reshape([10]*5+[2,2])
        // timeit np.linalg.inv(a)
        
        let shape = [10, 10, 10, 10, 10, 2, 2]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            _ = try! inv(a)
        }
    }
    
    func testInv2() {
        
        // a = np.arange(10**5*2*2, dtype=np.float32).reshape([10]*5+[2,2]).transpose([0, 1, 2, 3, 4, 6, 5])
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
    func testStack() {
        
        // a = np.arange(10**5, dtype=np.float32).reshape([10]*5)
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
    func testsClipped() {
        
        // a = np.arange(10**7, dtype=np.float32).reshape([10]*7).transpose()
        // timeit np.clip(a, 100, 1000)
        
        let shape = [10, 10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()
        measure {
            _ = a.clipped(low: 100, high: 1000)
        }
    }
    
    func testsMaximum() {
        
        // a = np.arange(10**6, dtype=np.float32).reshape([10]*6)
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
#endif
