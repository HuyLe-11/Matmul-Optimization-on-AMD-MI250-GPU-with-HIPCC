#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cmath>
//Time to execute on GPU: 0.824 ms 
// Performance: 2605.154541 GFLOP/s
// local block size = 32
// Block Tiling using 2D block tiling optimization
__host__ void matrix_initalize(float *a, float *b, float *c, int M, int N, int K)
{
    for (int i = 0; i < M * K; ++i)
    {
        a[i] = rand() % 10;
    }

    for (int i = 0; i < K * N; ++i)
    {
        b[i] = rand() % 10;
    }
}

__host__ void verify_result(float *a, float *b, float *c, int M, int N, int K)
{
    // int cnt = 0;
    float tmp = 0;

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            tmp = 0;
            for (int k = 0; k < K; ++k)
            {
                tmp += a[i * K + k] * b[k * N + j];
            }
            if (abs(tmp - c[i * N + j]) > 0.000001)
            {
                printf("check = %.2f and result = %.2f \n", tmp, c[i * N + j]);
                return;
            } 
            

        }
    }
    printf("Matrix multiplication is correct !\n");
}

__global__ void matmul_check(const float *d_a, const float *d_b, float *d_c, int M, int N, int K)
{
    const int bM = 64;
    const int bN = 64;
    const int bK = 64;

    // move blocktile to beginning of A's row and B's column
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    d_a += cRow * bM * K;
    d_b += cCol * bN;
    d_c += cRow * bM * N + cCol * bN;

    // The total shared memory used is (bM * bK * 4 (bytes) + bK * bN * 4 (bytes))
    __shared__ float As[bM * bK];
    __shared__ float Bs[bK * bN];

    // At thread level
    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;



    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += bK)
    {

        As[threadRow * bK + threadCol] = d_a[threadRow * K + threadCol];
        Bs[threadRow * bN + threadCol] = d_b[threadRow * N + threadCol];

        __syncthreads();

        d_a += bK;
        d_b += bK * N;


        for (int dotIdx = 0; dotIdx < bK; dotIdx++)
        {
            tmp += As[threadRow * bK + dotIdx] * Bs[dotIdx * bN + threadCol];
        }
        __syncthreads();
    }

    d_c[threadRow * N + threadCol] = tmp;
}

__global__ void matmul(const float *d_a, const float *d_b, float *d_c, int M, int N, int K){
    const int lcM = 32;
    const int lcN = 32;
    const int lcK = 32;


    const int bigRow = blockIdx.y;
    const int bigCol = blockIdx.x;

    d_a += bigRow * K * lcM;
    d_b += bigCol * lcN;
    d_c += bigRow * N * lcM + bigCol * lcN;

    __shared__ float l_a[lcM * lcK];
    __shared__ float l_b[lcK * lcN];

    const int lcRow = threadIdx.y;
    const int lcCol = threadIdx.x;

    float tmp = 0.0;
    for (int bl = 0; bl < K; bl += lcK){
        l_a[lcRow * lcK + lcCol] = d_a[lcRow * K + lcCol];
        l_b[lcRow * lcN + lcCol] = d_b[lcRow * N + lcCol];

        __syncthreads();

        d_a += lcK;
        d_b += N * lcK;

        for (int k = 0; k < lcK; k++){
            tmp += l_a[lcRow * lcK + k] * l_b[k * lcN + lcCol];
        }

        __syncthreads();
    }

    d_c[lcRow * N + lcCol] = tmp;
}

__host__ void execute(const float *a, const float *b, float *c, int M, int N, int K)
{
    int BLOCK_SIZE = 32;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(N / BLOCK_SIZE), ceil(M / BLOCK_SIZE));

    hipLaunchKernelGGL(matmul, dimGrid, dimBlock, 0, 0, a, b, c, M, N, K);
    (void)hipDeviceSynchronize();
}

int main()
{
    printf("1D block matrix matmul !! no optimization!! \n");

    int M = 1024;
    int N = 1024;
    int K = 1024;

    float *a;
    float *b;
    float *c;

    a = (float *)malloc(M * K * sizeof(float));
    b = (float *)malloc(K * N * sizeof(float));
    c = (float *)malloc(M * N * sizeof(float));

    matrix_initalize(a, b, c, M, N, K);

    float *d_a;
    float *d_b;
    float *d_c;

    (void)hipMalloc((void **)&d_a, M * K * sizeof(float));
    (void)hipMalloc((void **)&d_b, K * N * sizeof(float));
    (void)hipMalloc((void **)&d_c, M * N * sizeof(float));

    (void)hipMemcpy(d_a, a, M * K * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_b, b, K * N * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_c, c, M * N * sizeof(float), hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    (void)hipEventRecord(start, 0);

    execute(d_a, d_b, d_c, M, N, K);

    (void)hipEventRecord(stop, 0);
    (void)hipEventSynchronize(stop);
    float elapsedTime;
    (void)hipEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to execute on GPU: %.3f ms \n", elapsedTime);
    printf("Performance: %f GFLOP/s\n", 2.0f * M * N * K / (elapsedTime * 1e-3f * 1e9f));

    (void)hipMemcpy(c, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);

    verify_result(a, b, c, M, N, K);

    free(a);
    free(b);
    free(c);
    (void)hipFree(d_a);
    (void)hipFree(d_b);
    (void)hipFree(d_c);

    return 0;
}