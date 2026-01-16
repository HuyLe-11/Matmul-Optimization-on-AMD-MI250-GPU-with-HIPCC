#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstdlib>
// Time taken for matrix multiplication: 1.713 ms
// Performance: 1253.787 GFLOP/s
__global__ void matmul_gpu (const float *A, const float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__host__ void verifyresult(const float *A, const float *B, const float *C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                expected += A[i * K + k] * B[k * N + j];
            }
            if (fabs(C[i * N + j] - expected) > 1e-5)
            {
                printf("Mismatch at (%d, %d): got %.6f, expected %.6f\n", i, j, C[i * N + j], expected);
                return;
            }
        }
    }
    printf("Result verification passed!\n");
}


__host__ void matmul(const float *d_A, const float *d_B, float *d_C, int M, int N, int K){
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);    
    hipLaunchKernelGGL(matmul_gpu, gridSize, blockSize, 0, 0, d_A, d_B, d_C, M, N, K);
    (void)hipDeviceSynchronize();
}

int main(){
    int M, N, K;
    M = 1024;
    N = 1024;
    K = 1024;

    float *A, *B, *C;
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = rand() % 100;
    }
    for (int i = 0; i < K * N; ++i)
    {
        B[i] = rand() % 100;
    }
    
    float *d_A, *d_B, *d_C;
    (void)hipMalloc(&d_A, M * K * sizeof(float));
    (void)hipMalloc(&d_B, K * N * sizeof(float));
    (void)hipMalloc(&d_C, M * N * sizeof(float));

    (void)hipMemcpy(d_A, A, M * K * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_C, C, M * N * sizeof(float), hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    (void)hipEventRecord(start, 0);
    matmul(d_A, d_B, d_C, M, N, K);
    (void)hipEventRecord(stop, 0);
    (void)hipEventSynchronize(stop);
    float elapsedTime;
    (void)hipEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken for matrix multiplication: %.3f ms\n", elapsedTime);
    printf("Performance: %.3f GFLOP/s\n", 2.0f * N * N * N / (elapsedTime * 1e6));


    
    (void)hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);
    verifyresult(A, B, C, M, N, K);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);

    return 0;
}