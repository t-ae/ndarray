
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
}
#endif
