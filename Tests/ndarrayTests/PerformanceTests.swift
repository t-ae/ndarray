
import XCTest
import NDArray

class PerformanceTests: XCTestCase {

    func testAddPerformance1() {
        // two continuous arrays
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure { // 0.003sec
            // In [12]: a = np.arange(10**6).reshape([10]*6).astype(float)
            // In [13]: timeit a+a
            // 100 loops, best of 3: 2.73 ms per loop
            _ = a + a
        }
    }
    
    func testAddPerformance2() {
        // continuous + uncontinuous
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        measure { // 0.148sec
            // In [14]: a = np.arange(10**6).reshape([10]*6).astype(float)
            // In [15]: b = a.transpose()
            // In [16]: timeit a+b
            // 100 loops, best of 3: 13.4 ms per loop
            _ = a + b
        }
    }
    
    func testAddPerformance3() {
        // unconrinuous + uncontinuous
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        let b = a.transposed()
        let c = a.transposed(axes: [1, 2, 3, 4, 5, 0])
        measure { // 0.284sec
            // In [8]: a = np.arange(10**6).reshape([10]*6).astype(float)
            // In [9]: b = a.transpose()
            // In [11]: c = a.transpose([1,2,3,4,5,0])
            // In [12]: timeit b+c
            // 100 loops, best of 3: 5.95 ms per loop
            _ = b + c
        }
    }
    
    func testNegPerformance1() {
        // dense
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            // In [17]: a = np.arange(10**6).reshape([10]*6).astype(float)
            // In [18]: timeit (-a)
            // 100 loops, best of 3: 2.83 ms per loop
            _ = -a
        }
    }
    
    func testNegPerformance2() {
        // not dense
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()[1]
        measure {
            // In [15]: a = np.arange(10**6).reshape([10]*6).astype(float).transpose()[1]
            // In [16]: timeit (-a)
            // 1000 loops, best of 3: 573 µs per loop
            _ = -a
        }
    }
    
    func testSqrtPerformance1() {
        // dense
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape)
        measure {
            // In [2]: a = np.arange(10**6).reshape([10]*6)
            // In [3]: timeit np.sqrt(a)
            // 100 loops, best of 3: 4.86 ms per loop
            _ = sqrt(a)
        }
    }
    
    func testSqrtPerformance2() {
        // not dense
        let shape = [10, 10, 10, 10, 10, 10]
        let a = NDArray.range(shape.reduce(1, *)).reshaped(shape).transposed()[1]
        measure {
            // In [4]: a = np.arange(10**6).reshape([10]*6).transpose()[1]
            // In [5]: timeit np.sqrt(a)
            // 1000 loops, best of 3: 824 µs per loop
            _ = sqrt(a)
        }
    }
}
