
import XCTest
import Accelerate

#if !SWIFT_PACKAGE
class AcceleratePerformanceTests: XCTestCase {
    
    func testAdd_BLAS() {
        let count = 10_000_000
        let a = [Float](repeating: 1, count: count)
        let b = [Float](repeating: 1, count: count)
        
        measure {
            var ans = b
            ans.withUnsafeMutableBufferPointer { p in
                cblas_saxpy(Int32(count), 1, a, 1, p.baseAddress!, 1)
            }
        }
    }

    func testAdd_vDSP() {
        let count = 10_000_000
        let a = [Float](repeating: 1, count: count)
        let b = [Float](repeating: 1, count: count)
        measure {
            var ans = [Float](repeating: 1, count: count)
            vDSP_vadd(a, 1, b, 1, &ans, 1, vDSP_Length(count))
        }
    }
    
    func testMatmul_BLAS() {
        let M: Int32 = 1000
        let N: Int32 = 1000
        let K: Int32 = 1000
        let a = [Float](repeating: 1, count: Int(M*K))
        let b = [Float](repeating: 1, count: Int(K*N))
        var c = [Float](repeating: 0, count: Int(M*N))

        measure {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, M, b, K, 0, &c, M)
        }
    }

    func testMatmul_vDSP() {
        let M: vDSP_Length = 1000
        let N: vDSP_Length = 1000
        let K: vDSP_Length = 1000
        let a = [Float](repeating: 1, count: Int(M*K))
        let b = [Float](repeating: 1, count: Int(K*N))
        var c = [Float](repeating: 0, count: Int(M*N))
        
        measure {
            vDSP_mmul(a, 1, b, 1, &c, 1, M, N, K)
        }
    }
    
    func testCopy() {
        let stride = 2
        let c = 100_000_000
        let a = [Float](repeating: 1, count: c)
        var b = [Float](repeating: 0, count: c/stride)
        
        measure {
            for i in 0..<c/stride {
                b[i] = a[stride*i]
            }
        }
    }
    
    func testCopy_Pointer() {
        let stride = 2
        let c = 100_000_000
        let a = [Float](repeating: 1, count: c)
        var b = [Float](repeating: 0, count: c/stride)
        
        measure {
            var pa = UnsafePointer(a)
            var pb = UnsafeMutablePointer(mutating: &b)
            for _ in 0..<c/stride {
                pb.pointee = pa.pointee
                pa += stride
                pb += 1
            }
        }
    }
    
    func testCopy_BLAS() {
        let stride = 2
        let c = 100_000_000
        let a = [Float](repeating: 1, count: c)
        var b = [Float](repeating: 0, count: c/stride)
        measure {
            cblas_scopy(Int32(c/stride), a, Int32(stride), &b, 1)
        }
    }
    
    func testCopy_vDSP() {
        let stride = 2
        let c = 100_000_000
        let a = [Float](repeating: 1, count: c)
        var b = [Float](repeating: 0, count: c/stride)
        measure {
            vDSP_mmov(a, &b, vDSP_Length(c/stride), 1, vDSP_Length(stride), 1)
        }
    }
    
    func testCopyMinus_BLAS() {
        let stride = -4
        let c = 100_000_000
        let a = [Float](repeating: 1, count: c)
        var b = [Float](repeating: 0, count: -c/stride)
        measure {
            cblas_scopy(Int32(-c/stride), a, Int32(stride), &b, 1)
        }
    }
    
    func testCopyMinus_vDSP() {
        let stride = -4
        let c = 100_000_000
        let a = [Float](repeating: 1, count: c)
        var b = [Float](repeating: 0, count: -c/stride)
        measure {
            let src = UnsafePointer(a) - stride - 1
            vDSP_mmov(src, &b, vDSP_Length(-c/stride), 1, vDSP_Length(-stride), 1)
            vDSP_vrvrs(&b, 1, vDSP_Length(-c/stride))
        }
    }
    
    func testCopyMatrix_BLAS() {
        let m = 100000
        let n = 1000
        let a = [Float](repeating: 1, count: m*n)
        var b = [Float](repeating: 0, count: m*n/4)
        measure {
            let numInRow = Int32(n/2)
            var src = UnsafePointer(a)
            var dst = UnsafeMutablePointer(mutating: &b)
            let n2 = n/2
            for _ in 0..<m/2 {
                cblas_scopy(numInRow, src, 1, dst, 1)
                src += n
                dst += n2
            }
        }
    }
    
    func testCopyMatrix_vDSP() {
        let m = 100000
        let n = 1000
        let a = [Float](repeating: 1, count: m*n)
        var b = [Float](repeating: 0, count: m*n/4)
        measure {
            let n2 = n/2
            vDSP_mmov(a, &b, vDSP_Length(m/2), vDSP_Length(n/2), vDSP_Length(n), vDSP_Length(n2))
        }
    }
    
    func testAddA() {
        let count = 10_000_000
        let a = [Float](repeating: 0, count: count)
        let b: [Float] = [1]
        var ans = [Float](repeating: 0, count: count)
        measure {
            vDSP_vadd(a, 1, b, 0, &ans, 1, vDSP_Length(count))
        }
    }
    
    func testAddB() {
        let count = 10_000_000
        let a = [Float](repeating: 0, count: count)
        let b: [Float] = [1]
        var ans = [Float](repeating: 0, count: count)
        measure {
            vDSP_vsadd(a, 1, b, &ans, 1, vDSP_Length(count))
        }
    }
}
#endif
