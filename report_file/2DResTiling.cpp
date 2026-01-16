#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cassert>


template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_kernel_2d(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
    const int threadCol = threadIdx.x; // BN / TN
    const int threadRow = threadIdx.x; // BM / TM (thực tế là threadIdx.y trong dim3)
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const uint innerRowA = tid / BK;
    const uint innerColA = tid % BK;
    const uint strideA = numThreads / BK;

    const uint innerRowB = tid / BN;
    const uint innerColB = tid % BN;
    const uint strideB = numThreads / BN;

    T threadResults[TM * TN] = {0.0};
    T regM[TM];
    T regN[TN];

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK; 
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) regM[i] = As[(threadIdx.y * TM + i) * BK + dotIdx];
            for (uint i = 0; i < TN; ++i) regN[i] = Bs[dotIdx * BN + threadIdx.x * TN + i];
            
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadIdx.y * TM + resIdxM) * N + threadIdx.x * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
        }
    }
}


template<typename T>
void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
    printf("Verifying results...\n");
    for (int i = 0; i < std::min(M, 1024); i++) { // Kiểm tra giới hạn để tránh quá lâu
        for (int j = 0; j < std::min(N, 1024); j++) {
            T sum = 0;
            for (int k = 0; k < K; k++) sum += h_a[i * K + k] * h_b[k * N + j];
            if (std::abs(h_c[i * N + j] - sum) > 1e-2) {
                printf("Error at (%d,%d): CPU=%f, GPU=%f\n", i, j, (float)sum, (float)h_c[i * N + j]);
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

    printf("AMD MI250 GEMM (2D Register Tiling)\n");
    printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);

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

    const uint BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    dim3 block(BN / TN, BM / TM); // (16, 16) = 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // --- Warm-up ---
    printf("Warming up GPU...\n");
    for(int i=0; i<3; i++) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kernel_2d<float, BM, BN, BK, TM, TN>), 
                           grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
    }
    hipDeviceSynchronize();

    // --- Benchmark ---
    printf("Benchmarking...\n");
    hipEventRecord(start);
    
    const int iterations = 10;
    for(int i=0; i<iterations; i++) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kernel_2d<float, BM, BN, BK, TM, TN>), 
                           grid, block, 0, 0, d_a, d_b, d_c, M, N, K);
    }
    
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;
    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (avg_ms * 1e-3);

    printf("Average Time: %.3f ms\n", avg_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    hipMemcpy(h_c, d_c, size_c, hipMemcpyDeviceToHost);
    verifyResult(h_a, h_b, h_c, M, N, K);

    free(h_a); free(h_b); free(h_c);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    hipEventDestroy(start); hipEventDestroy(stop);

    return 0;
}