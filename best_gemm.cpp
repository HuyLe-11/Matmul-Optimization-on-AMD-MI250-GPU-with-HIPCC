#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cassert>

// Avg Time: 0.377 ms | Performance: 5701.69 GFLOPS

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_vectorized_2d(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // Vectorized load indices for SMEM (LDS)
    const uint innerRowA = tid / (BK / 4);
    const uint innerColA = tid % (BK / 4);
    const uint innerRowB = tid / (BN / 4);
    const uint innerColB = tid % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Vectorized load from Global Memory to LDS
        reinterpret_cast<float4 *>(&As[innerRowA * BK + innerColA * 4])[0] =
            reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // Vectorized write back to Global Memory
    for (uint resIdxM = 0; resIdxM < TM; resIdxM++) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
            
            tmp.x = threadResults[resIdxM * TN + resIdxN];
            tmp.y = threadResults[resIdxM * TN + resIdxN + 1];
            tmp.z = threadResults[resIdxM * TN + resIdxN + 2];
            tmp.w = threadResults[resIdxM * TN + resIdxN + 3];

            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
        }
    }
}

// --- Verification and Host Utilities ---

template<typename T>
void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
    for (int i = 0; i < std::min(M, 512); i++) {
        for (int j = 0; j < std::min(N, 512); j++) {
            T sum = 0;
            for (int k = 0; k < K; k++) sum += h_a[i * K + k] * h_b[k * N + j];
            if (std::abs(h_c[i * N + j] - sum) > 1e-2) {
                printf("Fail at (%d,%d): CPU=%f, GPU=%f\n", i, j, (float)sum, (float)h_c[i * N + j]);
                exit(1);
            }
        }
    }
    printf("Verification Success!\n");
}

int main(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-m") M = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "-n") N = std::stoi(argv[++i]);
        if (std::string(argv[i]) == "-k") K = std::stoi(argv[++i]);
    }

    printf("Matrix A: %dx%d, Matrix B: %dx%d, Matrix C: %dx%d 1 iter\n", M, K, K, N, M, N);

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a = (float*)malloc(size_a);
    float *h_b = (float*)malloc(size_b);
    float *h_c = (float*)malloc(size_c);

    for (int i = 0; i < M * K; i++) h_a[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_b[i] = (float)(rand() % 10);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size_a);
    hipMalloc(&d_b, size_b);
    hipMalloc(&d_c, size_c);

    hipMemcpy(d_a, h_a, size_a, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size_b, hipMemcpyHostToDevice);

    const uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM); 
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warm-up
    for(int i=0; i<3; i++) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vectorized_2d<float, BM, BN, BK, TM, TN>), 
                           grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
    }
    hipDeviceSynchronize();

    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    
    const int iter = 1;
    for(int i=0; i<iter; i++) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vectorized_2d<float, BM, BN, BK, TM, TN>), 
                           grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
    }
    
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);
    printf("Avg Time: %.3f ms | Performance: %.2f GFLOPS\n", ms/iter, (2.0*M*N*K*iter)/(ms*1e6));

    hipMemcpy(h_c, d_c, size_c, hipMemcpyDeviceToHost);
    verifyResult(h_a, h_b, h_c, M, N, K);

    free(h_a); free(h_b); free(h_c);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return 0;
}