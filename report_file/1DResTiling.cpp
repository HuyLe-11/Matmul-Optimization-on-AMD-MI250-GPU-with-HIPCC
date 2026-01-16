#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cmath>

// Kích thước Tile
#define BM 64  // Số hàng của C trong một Block
#define BN 64  // Số cột của C trong một Block
#define BK 8   // Độ sâu của Tile (K)
#define TM 8   // Số phần tử mỗi thread tính toán (theo chiều M)

__host__ void matrix_initalize(float *a, float *b, float *c, int M, int N, int K) {
    for (int i = 0; i < M * K; ++i) a[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; ++i) b[i] = (float)(rand() % 10);
}

__host__ void verify_result(float *a, float *b, float *c, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float tmp = 0;
            for (int k = 0; k < K; ++k) {
                tmp += a[i * K + k] * b[k * N + j];
            }
            if (std::abs(tmp - c[i * N + j]) > 0.1f) {
                printf("Lỗi tại [%d,%d]: check = %.2f, result = %.2f\n", i, j, tmp, c[i * N + j]);
                return;
            }
        }
    }
    printf("Matrix multiplication is correct!\n");
}

__global__ void matmul_1d_tiling(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tx = threadIdx.x; 
    const int ty = threadIdx.y; 

   
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    
    const int threadRow = ty * TM;


    float threadResults[TM];
    for (int i = 0; i < TM; i++) threadResults[i] = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        
       
        
        int tid = ty * blockDim.x + tx; 

        int a_row = tid / BK;
        int a_col = tid % BK;
        As[a_row * BK + a_col] = A[(by * BM + a_row) * K + (bkIdx + a_col)];

        int b_row = tid / BN;
        int b_col = tid % BN;
        Bs[b_row * BN + b_col] = B[(bkIdx + b_row) * N + (bx * BN + b_col)];

        __syncthreads();

        
        for (int k = 0; k < BK; k++) {
            float b_val = Bs[k * BN + tx];
            for (int i = 0; i < TM; i++) {
                threadResults[i] += As[(threadRow + i) * BK + k] * b_val;
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        C[(by * BM + threadRow + i) * N + (bx * BN + tx)] = threadResults[i];
    }
}

__host__ void execute(const float *d_a, const float *d_b, float *d_c, int M, int N, int K) {
    
    dim3 dimBlock(BN, BM / TM); 
    dim3 dimGrid(N / BN, M / BM);

    hipLaunchKernelGGL(matmul_1d_tiling, dimGrid, dimBlock, 0, 0, d_a, d_b, d_c, M, N, K);
    hipDeviceSynchronize();
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *a = (float *)malloc(size_A);
    float *b = (float *)malloc(size_B);
    float *c = (float *)malloc(size_C);

    matrix_initalize(a, b, c, M, N, K);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size_A);
    hipMalloc(&d_b, size_B);
    hipMalloc(&d_c, size_C);

    hipMemcpy(d_a, a, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, size_B, hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    execute(d_a, d_b, d_c, M, N, K);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, start, stop);

    printf("Time: %.3f ms | Perf: %.2f GFLOP/s\n", 
            elapsedTime, (2.0f * M * N * K) / (elapsedTime * 1e-3f * 1e9f));

    hipMemcpy(c, d_c, size_C, hipMemcpyDeviceToHost);
    verify_result(a, b, c, M, N, K);

    free(a); free(b); free(c);
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return 0;
}