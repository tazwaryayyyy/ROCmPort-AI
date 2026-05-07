import { useState, useEffect, useRef } from 'react'

const API_BASE = window.location.protocol === 'file:'
    ? 'http://localhost:8000'
    : window.location.origin

// ─── Global CSS ───────────────────────────────────────────────────────────────
const globalCSS = `
@import url('https://fonts.bunny.net/css?family=clash-display:400,500,600,700');
* { cursor: none !important; box-sizing: border-box; }
body { background: #080808; margin: 0; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080808; }
::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #b8ff57; }
.clash-display { font-family: 'Clash Display', sans-serif; }
@keyframes probe-blink { 0%,49%{opacity:1} 50%,100%{opacity:0} }
@keyframes badge-spin { to { transform: rotate(360deg) } }
@keyframes shine-sweep { from{background-position:200% 0} to{background-position:-200% 0} }
@keyframes slide-up { from{transform:translateY(100%);opacity:0} to{transform:translateY(0);opacity:1} }
@keyframes rocm-pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }
@keyframes nav-dot-pulse { 0%,100%{opacity:1;box-shadow:0 0 6px #ff4d00} 50%{opacity:0.5;box-shadow:0 0 2px #ff4d00} }
@keyframes seg-glow { 0%,100%{box-shadow:4px 0 10px rgba(255,77,0,0.6)} 50%{box-shadow:4px 0 18px rgba(255,77,0,0.9)} }
.benchmark-footer { animation: slide-up 300ms ease forwards; }
`

// ─── Template Kernels ─────────────────────────────────────────────────────────

const KERNEL_VECTOR_ADD = String.raw`
#include <cuda_runtime.h>
#include <stdio.h>

// Vector addition kernel with intentional warp size bug
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];

        // Intentional warp size bug - assumes 32 threads per warp
        // This will break on AMD wavefront (64 threads)
        if (threadIdx.x % 32 == 0) {
            // This synchronization only works for CUDA's 32-thread warps
            printf("Thread %d in warp %d completed\n", threadIdx.x, threadIdx.x / 32);
        }
    }
}

int main(void) {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            printf("Test FAILED at element %d!\n", i);
            break;
        }
    }
    printf("Test PASSED\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}
`.trim()

const KERNEL_MATRIX_MULTIPLY = String.raw`
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Matrix multiplication kernel with intentional warp size bug
// C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matrixMultiply(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;

        // Intentional warp size bug - assumes 32 threads per warp
        // This will cause incorrect behavior on AMD wavefront (64 threads)
        if (threadIdx.x % 32 == 0 && threadIdx.y % 32 == 0) {
            // This warp-level synchronization only works for CUDA
            printf("Block (%d,%d) warp (%d,%d) computed element (%d,%d) = %f\n",
                   blockIdx.x, blockIdx.y, threadIdx.x / 32, threadIdx.y / 32, row, col, sum);
        }
    }
}

// Optimized version with shared memory (for comparison)
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 31) / 32; ++tile) {
        if (row < M && tile * 32 + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tile * 32 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < N && tile * 32 + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    int M = 512, N = 512, K = 512;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_ref, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Matrix dimensions: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Warmup
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_ref, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_ref, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float basic_time = 0;
    cudaEventElapsedTime(&basic_time, start, stop);
    printf("Basic kernel time: %.3f ms\n", basic_time);

    cudaEventRecord(start);
    matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, stop);
    printf("Shared memory kernel time: %.3f ms\n", shared_time);
    printf("Speedup: %.2fx\n", basic_time / shared_time);

    cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            printf("Mismatch at element %d: %f != %f\n", i, h_C[i], h_C_ref[i]);
            correct = false;
            break;
        }
    }
    printf(correct ? "Verification PASSED\n" : "Verification FAILED\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);

    printf("Done\n");
    return 0;
}
`.trim()

const KERNEL_CONVOLUTION_2D = String.raw`
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 2D Convolution kernel with intentional warp size bug
__global__ void convolution2D(const float *input, const float *kernel, float *output,
                               int input_height, int input_width, int kernel_size,
                               int output_height, int output_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width) {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            for (int j = -kernel_radius; j <= kernel_radius; j++) {
                int input_row = row + i;
                int input_col = col + j;
                if (input_row >= 0 && input_row < input_height &&
                    input_col >= 0 && input_col < input_width) {
                    int kernel_row = i + kernel_radius;
                    int kernel_col = j + kernel_radius;
                    sum += input[input_row * input_width + input_col] *
                           kernel[kernel_row * kernel_size + kernel_col];
                }
            }
        }
        output[row * output_width + col] = sum;

        // Intentional warp size bug - assumes 32 threads per warp
        // This will break on AMD wavefront (64 threads)
        if (threadIdx.x % 32 == 0 && threadIdx.y % 32 == 0) {
            printf("Warp (%d,%d) processed output pixel (%d,%d) = %f\n",
                   threadIdx.x / 32, threadIdx.y / 32, row, col, sum);
        }
    }
}

// Shared memory version for comparison
__global__ void convolution2DShared(const float *input, const float *kernel, float *output,
                                    int input_height, int input_width, int kernel_size,
                                    int output_height, int output_width) {
    __shared__ float shared_input[32 + 6][32 + 6]; // +6 for 3x3 kernel padding
    __shared__ float shared_kernel[7][7];           // Max 7x7 kernel

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int kernel_radius = kernel_size / 2;

    if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
        shared_kernel[threadIdx.y][threadIdx.x] =
            kernel[threadIdx.y * kernel_size + threadIdx.x];
    }

    int input_row = blockIdx.y * blockDim.y + threadIdx.y - kernel_radius;
    int input_col = blockIdx.x * blockDim.x + threadIdx.x - kernel_radius;

    if (input_row >= 0 && input_row < input_height &&
        input_col >= 0 && input_col < input_width) {
        shared_input[threadIdx.y][threadIdx.x] =
            input[input_row * input_width + input_col];
    } else {
        shared_input[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (row < output_height && col < output_width) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++)
            for (int j = 0; j < kernel_size; j++)
                sum += shared_input[threadIdx.y + i][threadIdx.x + j] * shared_kernel[i][j];
        output[row * output_width + col] = sum;
    }
}

int main(int argc, char **argv) {
    int input_height = 1024, input_width = 1024, kernel_size = 3;
    int output_height = input_height - kernel_size + 1;
    int output_width  = input_width  - kernel_size + 1;

    size_t input_size        = input_height * input_width * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    size_t output_size       = output_height * output_width * sizeof(float);

    printf("Input: %dx%d, Kernel: %dx%d, Output: %dx%d\n",
           input_height, input_width, kernel_size, kernel_size, output_height, output_width);

    float *h_input      = (float *)malloc(input_size);
    float *h_kernel     = (float *)malloc(kernel_size_bytes);
    float *h_output     = (float *)malloc(output_size);
    float *h_output_ref = (float *)malloc(output_size);

    for (int i = 0; i < input_height * input_width; i++)
        h_input[i] = rand() / (float)RAND_MAX;

    float kernel_3x3[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    for (int i = 0; i < kernel_size * kernel_size; i++)
        h_kernel[i] = kernel_3x3[i];

    float *d_input, *d_kernel, *d_output, *d_output_ref;
    cudaMalloc(&d_input,      input_size);
    cudaMalloc(&d_kernel,     kernel_size_bytes);
    cudaMalloc(&d_output,     output_size);
    cudaMalloc(&d_output_ref, output_size);

    cudaMemcpy(d_input,  h_input,  input_size,        cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((output_width  + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Warmup
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_kernel, d_output_ref,
        input_height, input_width, kernel_size, output_height, output_width);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_kernel, d_output_ref,
        input_height, input_width, kernel_size, output_height, output_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float basic_time = 0;
    cudaEventElapsedTime(&basic_time, start, stop);
    printf("Basic kernel time: %.3f ms\n", basic_time);

    cudaEventRecord(start);
    convolution2DShared<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_kernel, d_output,
        input_height, input_width, kernel_size, output_height, output_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, stop);
    printf("Shared memory kernel time: %.3f ms\n", shared_time);
    printf("Speedup: %.2fx\n", basic_time / shared_time);

    cudaMemcpy(h_output_ref, d_output_ref, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output,     d_output,     output_size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < 100 && i < output_height * output_width; i++) {
        if (fabs(h_output[i] - h_output_ref[i]) > 1e-5) {
            printf("Mismatch at element %d: %f != %f\n", i, h_output[i], h_output_ref[i]);
            correct = false;
            break;
        }
    }
    printf(correct ? "Verification PASSED (first 100 elements)\n" : "Verification FAILED\n");

    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output); cudaFree(d_output_ref);
    free(h_input); free(h_kernel); free(h_output); free(h_output_ref);

    printf("Done\n");
    return 0;
}
`.trim()

const KERNEL_REDUCTION = String.raw`
#include <stdio.h>
#include <stdlib.h>

// compile: hipcc -arch=sm_60 -nocudalib reduction.cu

// --- IDE & COMPILER COMPATIBILITY LAYER ---
#if !defined(__CUDACC__) && !defined(__HIPCC__)
    #define __global__
    #define __shared__
    #define __syncthreads()
    struct dim3 {
        int x, y, z;
        dim3(int _x = 1, int _y = 1, int _z = 1) : x(_x), y(_y), z(_z) {}
    };
    typedef unsigned int cudaError_t;
    typedef void* cudaStream_t;
    dim3 threadIdx, blockIdx, blockDim;
    int warpSize = 64;
    #define cudaMalloc(p, s) (0)
    #define cudaFree(p) (0)
    #define cudaMemcpy(d, s, n, k) (0)
    #define cudaMemcpyHostToDevice 1
    #define cudaMemcpyDeviceToHost 2
    #define cudaSuccess 0
    #define cudaDeviceSynchronize() (0)
    #define LAUNCH_REDUCTION(g, b, m, ...) reduction_kernel(__VA_ARGS__)
#else
    #define LAUNCH_REDUCTION(g, b, m, ...) reduction_kernel<<<g, b, m>>>(__VA_ARGS__)
#endif
// ------------------------------------------

// Standard reduction template (first pass: block-level)
__global__ void reduction_kernel(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n)
        mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    // DELIBERATE WARP-SIZE BUG: Assuming warpSize=32 for final unrolled reduction
    // This will produce incorrect results on AMD (warpSize=64)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] = mySum = mySum + vsmem[tid + 32];
        vsmem[tid] = mySum = mySum + vsmem[tid + 16];
        vsmem[tid] = mySum = mySum + vsmem[tid + 8];
        vsmem[tid] = mySum = mySum + vsmem[tid + 4];
        vsmem[tid] = mySum = mySum + vsmem[tid + 2];
        vsmem[tid] = mySum = mySum + vsmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    const int N              = 1048576; // 1M elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);

    float *h_input  = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    LAUNCH_REDUCTION(blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float),
                     d_input, d_output, N);

    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) gpu_sum += h_output[i];
    float cpu_sum = (float)N;

    printf("Parallel Reduction (1M elements)\n");
    printf("CPU Sum: %.1f\n", cpu_sum);
    printf("GPU Sum: %.1f\n", gpu_sum);
    printf("Result: %s\n", (gpu_sum == cpu_sum) ? "PASS" : "FAIL (Warp size issue suspected)");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
`.trim()

// ─── Constants ────────────────────────────────────────────────────────────────

const TEMPLATES = {
    'Vector addition': KERNEL_VECTOR_ADD,
    'Matrix multiplication': KERNEL_MATRIX_MULTIPLY,
    '2D convolution': KERNEL_CONVOLUTION_2D,
    'Parallel reduction': KERNEL_REDUCTION,
}

const AGENT_LIST = ['analyzer', 'translator', 'optimizer', 'tester', 'coordinator']

const AGENT_LABEL = {
    analyzer: 'ANALYZER',
    translator: 'TRANSLATOR',
    optimizer: 'OPTIMIZER',
    tester: 'TESTER',
    coordinator: 'COORDINATOR',
}

const STATUS = {
    idle: {
        dot: '#1a1a1a',
        label: 'IDLE',
        borderLeft: '2px solid #1a1a1a',
        cardShadow: 'none',
        badgeColor: null,
    },
    running: {
        dot: '#ff4d00',
        label: 'RUNNING',
        borderLeft: '2px solid #ff4d00',
        cardShadow: '-4px 0 12px rgba(255,77,0,0.15)',
        badgeColor: 'orange',
    },
    done: {
        dot: '#b8ff57',
        label: 'DONE',
        borderLeft: '2px solid #b8ff57',
        cardShadow: 'none',
        badgeColor: 'green',
    },
    failed: {
        dot: '#ff3366',
        label: 'FAILED',
        borderLeft: '2px solid #ff3366',
        cardShadow: 'none',
        badgeColor: 'coral',
    },
}

const INITIAL_AGENTS = Object.fromEntries(
    AGENT_LIST.map(a => [a, { status: 'idle', message: 'Waiting\u2026', detail: '' }])
)

// ─── Component 1: ROCmCursor ──────────────────────────────────────────────────

const BRACKET_STYLES = `
[data-bracket-state="default"] .bkt-corner { border-color: rgba(184,255,87,0.7); }
[data-bracket-state="default"] .bkt-tl { top:0; left:0; }
[data-bracket-state="default"] .bkt-tr { top:0; right:0; }
[data-bracket-state="default"] .bkt-bl { bottom:0; left:0; }
[data-bracket-state="default"] .bkt-br { bottom:0; right:0; }

[data-bracket-state="button"] .bkt-corner { border-color: rgba(255,77,0,1); }
[data-bracket-state="button"] .bkt-tl { top:6px; left:6px; }
[data-bracket-state="button"] .bkt-tr { top:6px; right:6px; }
[data-bracket-state="button"] .bkt-bl { bottom:6px; left:6px; }
[data-bracket-state="button"] .bkt-br { bottom:6px; right:6px; }

[data-bracket-state="input"] .bkt-corner { border-color: rgba(179,232,255,0.7); }
[data-bracket-state="input"] .bkt-tl { top:0; left:0; }
[data-bracket-state="input"] .bkt-tr { top:0; right:0; }
[data-bracket-state="input"] .bkt-bl { bottom:0; left:0; }
[data-bracket-state="input"] .bkt-br { bottom:0; right:0; }

[data-bracket-state="running"] .bkt-corner { border-color: rgba(255,77,0,0.9); }
[data-bracket-state="running"] .bkt-tl { top:0; left:0; }
[data-bracket-state="running"] .bkt-tr { top:0; right:0; }
[data-bracket-state="running"] .bkt-bl { bottom:0; left:0; }
[data-bracket-state="running"] .bkt-br { bottom:0; right:0; }

.bkt-corner {
  position: absolute;
  width: 8px;
  height: 8px;
  transition: top 180ms ease, left 180ms ease, right 180ms ease, bottom 180ms ease, border-color 150ms ease;
}
.bkt-tl { border-top: 1.5px solid; border-left: 1.5px solid; border-bottom: 1.5px solid transparent; border-right: 1.5px solid transparent; }
.bkt-tr { border-top: 1.5px solid; border-right: 1.5px solid; border-bottom: 1.5px solid transparent; border-left: 1.5px solid transparent; }
.bkt-bl { border-bottom: 1.5px solid; border-left: 1.5px solid; border-top: 1.5px solid transparent; border-right: 1.5px solid transparent; }
.bkt-br { border-bottom: 1.5px solid; border-right: 1.5px solid; border-top: 1.5px solid transparent; border-left: 1.5px solid transparent; }
`

function ROCmCursor({ running }) {
    const dotRef = useRef(null)
    const boxRef = useRef(null)
    const mouseRef = useRef({ x: -200, y: -200 })
    const lerpRef = useRef({ x: -200, y: -200 })
    const rafRef = useRef(null)
    const targetTypeRef = useRef('default')
    const runningRef = useRef(running)

    useEffect(() => { runningRef.current = running }, [running])

    useEffect(() => {
        const onMove = (e) => {
            mouseRef.current = { x: e.clientX, y: e.clientY }
            if (dotRef.current) {
                dotRef.current.style.left = (e.clientX - 4) + 'px'
                dotRef.current.style.top = (e.clientY - 9) + 'px'
            }
        }

        const onOver = (e) => {
            const el = e.target
            if (el.closest('button, [role=button]')) {
                targetTypeRef.current = 'button'
            } else if (el.closest('textarea, input')) {
                targetTypeRef.current = 'input'
            } else {
                targetTypeRef.current = 'default'
            }
        }

        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseover', onOver)

        const loop = () => {
            const lx = lerpRef.current.x + (mouseRef.current.x - lerpRef.current.x) * 0.10
            const ly = lerpRef.current.y + (mouseRef.current.y - lerpRef.current.y) * 0.10
            lerpRef.current = { x: lx, y: ly }

            if (boxRef.current) {
                boxRef.current.style.left = (lx - 18) + 'px'
                boxRef.current.style.top = (ly - 18) + 'px'
                const state = runningRef.current ? 'running' : targetTypeRef.current
                boxRef.current.setAttribute('data-bracket-state', state)
            }

            rafRef.current = requestAnimationFrame(loop)
        }
        rafRef.current = requestAnimationFrame(loop)

        return () => {
            window.removeEventListener('mousemove', onMove)
            window.removeEventListener('mouseover', onOver)
            cancelAnimationFrame(rafRef.current)
        }
    }, [running])

    const blinkDuration = running ? '0.5s' : '1s'
    const dotColor = running ? '#ff4d00' : '#b8ff57'

    return (
        <>
            <style dangerouslySetInnerHTML={{ __html: BRACKET_STYLES }} />
            <span
                ref={dotRef}
                style={{
                    position: 'fixed',
                    zIndex: 9999,
                    pointerEvents: 'none',
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: '12px',
                    color: dotColor,
                    lineHeight: 1,
                    userSelect: 'none',
                    animation: `probe-blink ${blinkDuration} step-end infinite`,
                }}
            >
                &#9611;
            </span>
            <div
                ref={boxRef}
                data-bracket-state="default"
                style={{
                    position: 'fixed',
                    width: '36px',
                    height: '36px',
                    zIndex: 9998,
                    pointerEvents: 'none',
                }}
            >
                <div className="bkt-corner bkt-tl" />
                <div className="bkt-corner bkt-tr" />
                <div className="bkt-corner bkt-bl" />
                <div className="bkt-corner bkt-br" />
            </div>
        </>
    )
}

// ─── Component 2: AetherFlowHero ─────────────────────────────────────────────

function AetherFlowHero({ running, mousePos }) {
    const canvasRef = useRef(null)
    const particlesRef = useRef([])
    const rafRef = useRef(null)
    const runningRef = useRef(running)

    useEffect(() => {
        runningRef.current = running
    }, [running])

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')

        const COLORS = [
            '#b8ff57', '#b8ff57', '#b8ff57', '#b8ff57',
            '#ff4d00', '#ff4d00',
            '#b3e8ff', '#b3e8ff',
            '#ff3366',
            '#ffd60a',
        ]

        const resize = () => {
            canvas.width = window.innerWidth
            canvas.height = window.innerHeight
        }
        resize()

        const initParticles = () => {
            particlesRef.current = Array.from({ length: 80 }, () => ({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.6,
                vy: (Math.random() - 0.5) * 0.6,
                radius: 1 + Math.random(),
                opacity: 0.3 + Math.random() * 0.5,
                color: COLORS[Math.floor(Math.random() * COLORS.length)],
            }))
        }
        initParticles()

        let debounceTimer = null
        const onResize = () => {
            clearTimeout(debounceTimer)
            debounceTimer = setTimeout(() => { resize(); initParticles() }, 100)
        }
        window.addEventListener('resize', onResize)

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            const ps = particlesRef.current
            const isRunning = runningRef.current
            const mx = mousePos && mousePos.current ? mousePos.current.x : -999
            const my = mousePos && mousePos.current ? mousePos.current.y : -999
            const speedMult = isRunning ? 2.5 : 1
            const opacityBonus = isRunning ? 0.2 : 0
            const connMult = isRunning ? 3 : 1

            for (const p of ps) {
                const pdx = p.x - mx
                const pdy = p.y - my
                const dist = Math.sqrt(pdx * pdx + pdy * pdy)
                if (dist < 100 && dist > 0) {
                    const force = Math.min(1.5, 80 / dist)
                    p.vx += (pdx / dist) * force * 0.04
                    p.vy += (pdy / dist) * force * 0.04
                }

                const maxSpeed = 0.3 * speedMult
                const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy)
                if (spd > maxSpeed) {
                    p.vx = (p.vx / spd) * maxSpeed
                    p.vy = (p.vy / spd) * maxSpeed
                }

                p.x += p.vx * speedMult
                p.y += p.vy * speedMult

                if (p.x < 0) p.x += canvas.width
                if (p.x > canvas.width) p.x -= canvas.width
                if (p.y < 0) p.y += canvas.height
                if (p.y > canvas.height) p.y -= canvas.height

                const op = Math.min(1, p.opacity + opacityBonus)
                const hexOp = Math.round(op * 255).toString(16).padStart(2, '0')
                ctx.beginPath()
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
                ctx.fillStyle = p.color + hexOp
                ctx.fill()
            }

            for (let i = 0; i < ps.length; i++) {
                for (let j = i + 1; j < ps.length; j++) {
                    const cdx = ps[i].x - ps[j].x
                    const cdy = ps[i].y - ps[j].y
                    const d = Math.sqrt(cdx * cdx + cdy * cdy)
                    if (d < 120) {
                        const alpha = (1 - d / 120) * 0.15 * connMult
                        ctx.beginPath()
                        ctx.moveTo(ps[i].x, ps[i].y)
                        ctx.lineTo(ps[j].x, ps[j].y)
                        ctx.strokeStyle = `rgba(184,255,87,${Math.min(1, alpha)})`
                        ctx.lineWidth = 0.5
                        ctx.stroke()
                    }
                }
            }

            rafRef.current = requestAnimationFrame(draw)
        }
        rafRef.current = requestAnimationFrame(draw)

        return () => {
            cancelAnimationFrame(rafRef.current)
            window.removeEventListener('resize', onResize)
            clearTimeout(debounceTimer)
        }
    }, [])

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                inset: 0,
                zIndex: -1,
                pointerEvents: 'none',
                width: '100vw',
                height: '100vh',
            }}
        />
    )
}

// ─── Component 3: AnimatedShinyText ──────────────────────────────────────────

function AnimatedShinyText({ children, className }) {
    return (
        <span style={{ position: 'relative', display: 'inline-block' }} className={className}>
            {children}
            <span
                style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.18) 50%, transparent 60%)',
                    backgroundSize: '200% 100%',
                    animation: 'shine-sweep 2.4s linear infinite',
                    pointerEvents: 'none',
                    borderRadius: 'inherit',
                }}
            />
        </span>
    )
}

// ─── Component 4: HeroBadge ───────────────────────────────────────────────────

const BADGE_COLOR_MAP = {
    green: '#b8ff57',
    gold: '#ffd60a',
    orange: '#ff4d00',
    coral: '#ff3366',
    blue: '#b3e8ff',
}

function HeroBadge({ children, color }) {
    const c = BADGE_COLOR_MAP[color] || '#b8ff57'
    return (
        <span style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', padding: '1px', borderRadius: '3px', overflow: 'hidden', flexShrink: 0 }}>
            <span style={{
                position: 'absolute',
                inset: '-50%',
                width: '200%',
                height: '200%',
                background: `conic-gradient(from 0deg, transparent 0%, ${c} 20%, transparent 40%)`,
                animation: 'badge-spin 3s linear infinite',
                zIndex: 0,
            }} />
            <span style={{
                position: 'relative',
                zIndex: 1,
                background: '#0f0f0f',
                borderRadius: '2px',
                padding: '2px 8px',
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '10px',
                color: c,
                whiteSpace: 'nowrap',
            }}>
                {children}
            </span>
        </span>
    )
}

// ─── Component 5: ModernAnimatedHero ─────────────────────────────────────────

const HERO_WORDS = ['cudaMalloc', 'hipMalloc', '__global__', 'wavefront64', 'gfx942', 'rocprof', 'HIP', 'ROCm', 'LDS', 'vsmem', 'tid', 'blockDim', 'threadIdx', '__syncthreads', '0xFF', '0x3F', '\u223f', '\u2295', '\u2593', '\u2591']
const SCRAMBLE_POOL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\u223f\u2295\u2593\u2591'
const HERO_TARGET_STR = 'CUDA \u2192 ROCm / MI300X'

function ModernAnimatedHero() {
    const canvasRef = useRef(null)
    const rafRef = useRef(null)
    const [displayChars, setDisplayChars] = useState(
        () => HERO_TARGET_STR.split('').map(c => ({ char: c, scrambling: false }))
    )

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')

        const setSize = () => {
            canvas.width = canvas.offsetWidth || window.innerWidth
            canvas.height = 56
        }
        setSize()

        const NUM_COLS = 20
        const cols = Array.from({ length: NUM_COLS }, (_, i) => ({
            x: (canvas.width / NUM_COLS) * i + (canvas.width / NUM_COLS / 2),
            y: Math.random() * 56,
            speed: 0.4 + Math.random() * 0.8,
            wordIdx: Math.floor(Math.random() * HERO_WORDS.length),
        }))

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.font = '9px "JetBrains Mono", monospace'
            for (const col of cols) {
                const word = HERO_WORDS[col.wordIdx % HERO_WORDS.length]
                ctx.fillStyle = 'rgba(184,255,87,0.25)'
                ctx.fillText(word[0], col.x, col.y)
                col.y += col.speed
                if (col.y > 70) {
                    col.y = -10
                    col.wordIdx = (col.wordIdx + 1) % HERO_WORDS.length
                }
            }
            rafRef.current = requestAnimationFrame(draw)
        }
        rafRef.current = requestAnimationFrame(draw)
        return () => cancelAnimationFrame(rafRef.current)
    }, [])

    const scrambleAll = () => {
        const target = HERO_TARGET_STR.split('')
        target.forEach((finalChar, i) => {
            const delay = i * 30
            const duration = 600
            setTimeout(() => {
                const start = Date.now()
                const tick = setInterval(() => {
                    const elapsed = Date.now() - start
                    if (elapsed >= duration) {
                        clearInterval(tick)
                        setDisplayChars(prev => {
                            const next = [...prev]
                            next[i] = { char: finalChar, scrambling: false }
                            return next
                        })
                    } else {
                        setDisplayChars(prev => {
                            const next = [...prev]
                            next[i] = { char: SCRAMBLE_POOL[Math.floor(Math.random() * SCRAMBLE_POOL.length)], scrambling: true }
                            return next
                        })
                    }
                }, 40)
            }, delay)
        })
    }

    useEffect(() => {
        scrambleAll()
        const id = setInterval(scrambleAll, 8000)
        return () => clearInterval(id)
    }, [])

    return (
        <div style={{ position: 'relative', height: '56px', width: '100%', overflow: 'hidden', background: '#080808', borderBottom: '1px solid #1a1a1a', flexShrink: 0 }}>
            <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '56px' }} />
            <div style={{
                position: 'absolute', inset: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                pointerEvents: 'none',
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: '13px',
                fontWeight: 600,
                letterSpacing: '2px',
            }}>
                {displayChars.map((c, i) => (
                    <span key={i} style={{ color: c.scrambling ? '#555555' : '#b8ff57' }}>{c.char}</span>
                ))}
            </div>
        </div>
    )
}

// ─── Component 6: HeroDitheringCard ──────────────────────────────────────────

const BAYER_4X4 = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]

function HeroDitheringCard({ children, style: extraStyle }) {
    const canvasRef = useRef(null)
    const drawnRef = useRef(false)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas || drawnRef.current) return
        const w = Math.max(canvas.offsetWidth, 200)
        const h = Math.max(canvas.offsetHeight, 80)
        canvas.width = w
        canvas.height = h
        drawnRef.current = true
        const ctx = canvas.getContext('2d')
        const img = ctx.createImageData(w, h)
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const threshold = BAYER_4X4[y % 4][x % 4] / 16
                const idx = (y * w + x) * 4
                if (threshold > 0.5) {
                    img.data[idx] = 184; img.data[idx + 1] = 255; img.data[idx + 2] = 87; img.data[idx + 3] = 255
                } else {
                    img.data[idx] = 8; img.data[idx + 1] = 8; img.data[idx + 2] = 8; img.data[idx + 3] = 255
                }
            }
        }
        ctx.putImageData(img, 0, 0)
    }, [])

    return (
        <div style={{ position: 'relative', overflow: 'hidden', borderRadius: '2px', ...extraStyle }}>
            <canvas
                ref={canvasRef}
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 0, opacity: 0.06, pointerEvents: 'none' }}
            />
            <div style={{ position: 'relative', zIndex: 1 }}>
                {children}
            </div>
        </div>
    )
}

// ─── Component 7: WrapShader ──────────────────────────────────────────────────

let _wrapId = 0

function WrapShader({ children, active }) {
    const idRef = useRef(null)
    if (!idRef.current) idRef.current = `rocm-wrap-${++_wrapId}`
    const dispMapRef = useRef(null)
    const rafRef = useRef(null)
    const startRef = useRef(null)
    const activeRef = useRef(active)

    useEffect(() => { activeRef.current = active }, [active])

    useEffect(() => {
        if (!active) {
            cancelAnimationFrame(rafRef.current)
            if (dispMapRef.current) dispMapRef.current.setAttribute('scale', '3')
            return
        }
        startRef.current = performance.now()
        const animate = (now) => {
            if (!activeRef.current) {
                if (dispMapRef.current) dispMapRef.current.setAttribute('scale', '3')
                return
            }
            const t = ((now - startRef.current) % 2000) / 2000
            const scale = 3 + Math.sin(t * Math.PI * 2) * 3
            if (dispMapRef.current) dispMapRef.current.setAttribute('scale', String(scale.toFixed(2)))
            rafRef.current = requestAnimationFrame(animate)
        }
        rafRef.current = requestAnimationFrame(animate)
        return () => cancelAnimationFrame(rafRef.current)
    }, [active])

    const filterId = idRef.current

    return (
        <div style={{ position: 'relative', width: '42%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <svg style={{ position: 'absolute', width: 0, height: 0, overflow: 'visible' }} aria-hidden="true">
                <defs>
                    <filter id={filterId}>
                        <feTurbulence type="fractalNoise" baseFrequency="0.015 0.015" numOctaves="2" seed="8" result="noise" />
                        <feDisplacementMap ref={dispMapRef} in="SourceGraphic" in2="noise" scale="3" xChannelSelector="R" yChannelSelector="G" />
                    </filter>
                </defs>
            </svg>
            <div style={{ filter: `url(#${filterId})`, flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {children}
            </div>
        </div>
    )
}

// ─── MagneticButton ───────────────────────────────────────────────────────────

function MagneticButton({ children, onClick, disabled }) {
    const btnRef = useRef(null)
    const [transform, setTransform] = useState('translate(0px,0px)')
    const [isHover, setIsHover] = useState(false)

    const onMouseMove = (e) => {
        if (disabled) return
        const rect = btnRef.current.getBoundingClientRect()
        const cx = rect.left + rect.width / 2
        const cy = rect.top + rect.height / 2
        const dx = Math.max(-10, Math.min(10, (e.clientX - cx) * 0.35))
        const dy = Math.max(-10, Math.min(10, (e.clientY - cy) * 0.35))
        setTransform(`translate(${dx}px,${dy}px)`)
    }

    const onMouseLeave = () => {
        setTransform('translate(0px,0px)')
        setIsHover(false)
    }

    const onMouseEnter = () => { if (!disabled) setIsHover(true) }

    return (
        <button
            ref={btnRef}
            onClick={onClick}
            disabled={disabled}
            onMouseMove={onMouseMove}
            onMouseLeave={onMouseLeave}
            onMouseEnter={onMouseEnter}
            style={{
                width: '100%',
                height: '52px',
                background: disabled ? '#1a1a1a' : (isHover ? '#ff3300' : '#ff4d00'),
                color: disabled ? '#333333' : '#ffffff',
                border: 'none',
                borderRadius: '3px',
                cursor: disabled ? 'not-allowed' : 'pointer',
                transform: disabled ? 'translate(0px,0px)' : transform,
                transition: isHover
                    ? 'transform 100ms linear, box-shadow 150ms, background 150ms'
                    : 'transform 400ms cubic-bezier(0.34,1.56,0.64,1), box-shadow 150ms, background 150ms',
                boxShadow: !disabled && isHover ? '0 0 32px rgba(255,77,0,0.45)' : 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
            }}
        >
            {children}
        </button>
    )
}

// ─── AgentCard ────────────────────────────────────────────────────────────────

function AgentCard({ name, state }) {
    const s = STATUS[state.status] ?? STATUS.idle
    const isRunning = state.status === 'running'
    const isDoneCoord = state.status === 'done' && name === 'coordinator'

    return (
        <div style={{ background: 'transparent', padding: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{
                    flexShrink: 0,
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: s.dot,
                    display: 'inline-block',
                    animation: isRunning ? 'rocm-pulse 1.1s ease-in-out infinite' : 'none',
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '10px', color: '#555555', letterSpacing: '0.15em', textTransform: 'uppercase' }}>
                        {AGENT_LABEL[name]}
                    </div>
                    <div style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '13px', color: '#e8e8e8', marginTop: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {isDoneCoord
                            ? <AnimatedShinyText><span style={{ color: '#b8ff57' }}>{state.message || 'Waiting\u2026'}</span></AnimatedShinyText>
                            : (state.message || 'Waiting\u2026')}
                    </div>
                </div>
                {s.badgeColor && (
                    <HeroBadge color={s.badgeColor}>{s.label}</HeroBadge>
                )}
            </div>
            {state.detail && (
                <p style={{
                    marginTop: '8px',
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: '11px',
                    color: '#555555',
                    fontStyle: 'italic',
                    lineHeight: '1.5',
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    margin: '8px 0 0 0',
                }}>
                    {state.detail}
                </p>
            )}
        </div>
    )
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
    const [code, setCode] = useState('')
    const [activeTemplate, setActiveTemplate] = useState(null)
    const [agents, setAgents] = useState(INITIAL_AGENTS)
    const [running, setRunning] = useState(false)
    const [elapsed, setElapsed] = useState(0)
    const [benchmark, setBenchmark] = useState(null)
    const [errorBanner, setErrorBanner] = useState(null)

    const timerRef = useRef(null)
    const startRef = useRef(null)
    const mousePos = useRef({ x: -999, y: -999 })

    const lineCount = code ? code.split('\n').length : 1

    useEffect(() => {
        const onMove = (e) => { mousePos.current = { x: e.clientX, y: e.clientY } }
        window.addEventListener('mousemove', onMove)
        return () => window.removeEventListener('mousemove', onMove)
    }, [])

    // ── Timer ─────────────────────────────────────────────────────────────────
    const startTimer = () => {
        startRef.current = Date.now()
        timerRef.current = setInterval(
            () => setElapsed(Date.now() - startRef.current),
            100
        )
    }

    const stopTimer = () => {
        clearInterval(timerRef.current)
        timerRef.current = null
    }

    useEffect(() => () => stopTimer(), [])

    // ── Helpers ───────────────────────────────────────────────────────────────
    const resetAgents = () =>
        setAgents(Object.fromEntries(
            AGENT_LIST.map(a => [a, { status: 'idle', message: 'Waiting\u2026', detail: '' }])
        ))

    const updateAgent = (agent, patch) =>
        setAgents(prev => ({ ...prev, [agent]: { ...prev[agent], ...patch } }))

    const selectTemplate = (name) => {
        setActiveTemplate(name)
        setCode(TEMPLATES[name])
    }

    const fmtElapsed = (ms) => `${(ms / 1000).toFixed(1)}s`

    // ── Demo mode fallback ────────────────────────────────────────────────────
    const runDemo = async () => {
        const steps = [
            { agent: 'analyzer', status: 'running', message: 'Scanning CUDA patterns\u2026', detail: '' },
            { agent: 'analyzer', status: 'done', message: 'Found 3 critical AMD issues', detail: 'warp-32 assumption in reduction tail, threadIdx%32 idiom, LDS bank conflict pattern' },
            { agent: 'translator', status: 'running', message: 'Running hipify + LLM pass\u2026', detail: '' },
            { agent: 'translator', status: 'done', message: 'Translation complete', detail: 'hipify applied; 7 additional LLM corrections for wavefront-64 semantics' },
            { agent: 'optimizer', status: 'running', message: 'Proposing optimizations\u2026', detail: '' },
            { agent: 'optimizer', status: 'done', message: '4 optimization patches generated', detail: 'LDS padding, wavefront-aware reduction, coalesced access pattern' },
            { agent: 'tester', status: 'running', message: 'Compiling with hipcc\u2026', detail: '' },
            { agent: 'tester', status: 'done', message: 'Compiled and profiled on gfx942', detail: 'rocprof: 0.026 ms \u2014 correctness verified' },
            { agent: 'coordinator', status: 'running', message: 'Assembling final report\u2026', detail: '' },
            { agent: 'coordinator', status: 'done', message: 'Migration complete \u2014 2.61\u00d7 speedup', detail: 'data_source: demo_artifact' },
        ]

        for (const step of steps) {
            await new Promise(r => setTimeout(r, 800))
            updateAgent(step.agent, { status: step.status, message: step.message, detail: step.detail })
        }

        setBenchmark({
            total_changes: 11,
            bugs_found: 3,
            compiled_successfully: true,
            data_source: 'demo_artifact',
        })
        stopTimer()
        setRunning(false)
    }

    // ── Main action ───────────────────────────────────────────────────────────
    const handlePort = async () => {
        if (running || !code.trim()) return

        setRunning(true)
        setElapsed(0)
        setBenchmark(null)
        setErrorBanner(null)
        resetAgents()
        startTimer()

        try {
            const res = await fetch(`${API_BASE}/port`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    cuda_code: code,
                    kernel_name: activeTemplate || 'custom',
                    simple_mode: false,
                }),
            })
            if (!res.ok) throw new Error(`HTTP ${res.status}`)

            const reader = res.body.getReader()
            const dec = new TextDecoder()
            let buf = ''

            outer: while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buf += dec.decode(value, { stream: true })
                const lines = buf.split('\n')
                buf = lines.pop()

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    const raw = line.slice(6).trim()
                    if (raw === '[DONE]') break outer

                    try {
                        const ev = JSON.parse(raw)
                        if (!ev.agent) continue

                        updateAgent(ev.agent, {
                            status: ev.status,
                            message: ev.message ?? '',
                            detail: ev.detail ?? '',
                        })

                        if (ev.agent === 'coordinator' && ev.status === 'done') {
                            let report = ev.result ?? null
                            if (!report && ev.detail) {
                                try { report = JSON.parse(ev.detail) } catch (_) { report = null }
                            }
                            const r = report ?? ev
                            setBenchmark({
                                total_changes: r.total_changes ?? r.changes_made ?? '\u2014',
                                bugs_found: r.bugs_found ?? r.critical_bugs ?? r.static_risk_report?.critical_count ?? '\u2014',
                                compiled_successfully: r.compiled_successfully ?? r.compiled ?? r.migration_success ?? false,
                                data_source: r.data_source ?? 'unknown',
                            })
                        }
                    } catch (_) { /* malformed SSE line — skip */ }
                }
            }
        } catch {
            setErrorBanner('Backend unavailable \u2014 running in demo mode')
            runDemo()
            return
        }

        stopTimer()
        setRunning(false)
    }

    // ── Segment data ──────────────────────────────────────────────────────────
    const segLabels = ['ANA', 'TRA', 'OPT', 'TES', 'CORD']
    const segAgents = ['analyzer', 'translator', 'optimizer', 'tester', 'coordinator']

    const cardStyle = (status) => ({
        borderLeft: STATUS[status]?.borderLeft || '2px solid #1a1a1a',
        boxShadow: STATUS[status]?.cardShadow || 'none',
        background: '#0f0f0f',
    })

    // ── Render ────────────────────────────────────────────────────────────────
    return (
        <>
            <style dangerouslySetInnerHTML={{ __html: globalCSS }} />

            <ROCmCursor running={running} />
            <AetherFlowHero running={running} mousePos={mousePos} />

            <div style={{
                minHeight: '100vh',
                display: 'flex',
                flexDirection: 'column',
                background: 'rgba(8,8,8,0.82)',
                color: '#f0f0f0',
                fontFamily: '"JetBrains Mono", monospace',
            }}>

                {/* ── Error banner ─────────────────────────────────────────────── */}
                {errorBanner && (
                    <div style={{
                        flexShrink: 0,
                        padding: '8px 24px',
                        background: '#0c0000',
                        borderBottom: '1px solid #ff3366',
                        color: '#ff3366',
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: '12px',
                    }}>
                        \u26a0 {errorBanner}
                    </div>
                )}

                {/* ── NAVBAR ───────────────────────────────────────────────────── */}
                <nav style={{
                    flexShrink: 0,
                    height: '40px',
                    background: '#080808',
                    borderBottom: '1px solid #1a1a1a',
                    position: 'sticky',
                    top: 0,
                    zIndex: 50,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0 20px',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span className="clash-display" style={{ fontSize: '16px', fontWeight: 700, letterSpacing: '-0.02em', display: 'inline-flex', alignItems: 'center' }}>
                            <AnimatedShinyText>
                                <span style={{ color: '#f0f0f0' }}>ROCmPort</span>
                            </AnimatedShinyText>
                            <span style={{ color: '#b8ff57' }}>AI</span>
                        </span>
                        <HeroBadge color="gold">AMD MI300X</HeroBadge>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{
                            fontFamily: '"JetBrains Mono", monospace',
                            fontSize: '13px',
                            color: running ? '#b8ff57' : '#333333',
                            transition: 'color 0.3s',
                        }}>
                            {fmtElapsed(elapsed)}
                        </span>
                        <span style={{
                            width: '7px',
                            height: '7px',
                            borderRadius: '50%',
                            background: running ? '#ff4d00' : (benchmark ? '#b8ff57' : '#2a2a2a'),
                            animation: running ? 'nav-dot-pulse 1s ease-in-out infinite' : 'none',
                            display: 'inline-block',
                            flexShrink: 0,
                        }} />
                    </div>
                </nav>

                {/* ── HERO STRIP ───────────────────────────────────────────────── */}
                <ModernAnimatedHero />

                {/* ── TWO-COLUMN MAIN ──────────────────────────────────────────── */}
                <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

                    {/* ── LEFT PANEL 58% ─────────────────────────────────────────── */}
                    <div style={{
                        width: '58%',
                        display: 'flex',
                        flexDirection: 'column',
                        padding: '20px',
                        gap: '16px',
                        borderRight: '1px solid #1a1a1a',
                        overflowY: 'auto',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '11px', color: '#555555' }}>// CUDA source</span>
                            <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '11px', color: '#555555' }}>{lineCount} lines</span>
                        </div>

                        <textarea
                            value={code}
                            onChange={e => { setCode(e.target.value); setActiveTemplate(null) }}
                            placeholder={'// Paste CUDA code here\n// or pick a demo below'}
                            spellCheck={false}
                            style={{
                                width: '100%',
                                minHeight: '340px',
                                resize: 'vertical',
                                background: '#0c0c0c',
                                border: '1px solid #1a1a1a',
                                borderRadius: '4px',
                                padding: '16px',
                                color: '#e8e8e8',
                                fontFamily: '"JetBrains Mono", monospace',
                                fontSize: '13px',
                                lineHeight: '1.65',
                                caretColor: '#b8ff57',
                                tabSize: 4,
                                outline: 'none',
                                transition: 'border-color 0.15s, box-shadow 0.15s',
                            }}
                            onFocus={e => {
                                e.target.style.borderColor = '#b8ff57'
                                e.target.style.boxShadow = '0 0 0 2px rgba(184,255,87,0.08)'
                            }}
                            onBlur={e => {
                                e.target.style.borderColor = '#1a1a1a'
                                e.target.style.boxShadow = 'none'
                            }}
                        />

                        <div>
                            <p style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '11px', color: '#555555', marginBottom: '10px', marginTop: 0 }}>
                                Select a template:
                            </p>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                {Object.keys(TEMPLATES).map(name => (
                                    <button
                                        key={name}
                                        onClick={() => selectTemplate(name)}
                                        style={{
                                            border: activeTemplate === name ? '1px solid #b8ff57' : '1px solid #1a1a1a',
                                            background: activeTemplate === name ? 'rgba(184,255,87,0.06)' : '#0f0f0f',
                                            color: activeTemplate === name ? '#b8ff57' : '#888888',
                                            fontFamily: '"JetBrains Mono", monospace',
                                            fontSize: '12px',
                                            borderRadius: '4px',
                                            padding: '4px 12px',
                                            cursor: 'pointer',
                                            transition: 'border-color 0.15s, color 0.15s',
                                        }}
                                        onMouseEnter={e => {
                                            if (activeTemplate !== name) {
                                                e.currentTarget.style.borderColor = 'rgba(184,255,87,0.4)'
                                                e.currentTarget.style.color = 'rgba(184,255,87,0.6)'
                                            }
                                        }}
                                        onMouseLeave={e => {
                                            if (activeTemplate !== name) {
                                                e.currentTarget.style.borderColor = '#1a1a1a'
                                                e.currentTarget.style.color = '#888888'
                                            }
                                        }}
                                    >
                                        {name}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <MagneticButton onClick={handlePort} disabled={running || !code.trim()}>
                            <AnimatedShinyText>
                                <span style={{
                                    fontFamily: '"JetBrains Mono", monospace',
                                    fontSize: '14px',
                                    letterSpacing: '3px',
                                    textTransform: 'uppercase',
                                    fontWeight: 600,
                                }}>
                                    {running ? 'RUNNING\u2026' : 'PORT TO ROCM'}
                                </span>
                            </AnimatedShinyText>
                        </MagneticButton>
                    </div>

                    {/* ── RIGHT PANEL 42% (WrapShader owns the width) ─────────── */}
                    <WrapShader active={running}>
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            padding: '20px',
                            gap: '16px',
                            overflowY: 'auto',
                            height: '100%',
                            isolation: 'isolate',
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '11px', color: '#555555' }}>// Pipeline</span>
                                <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '12px', color: running ? '#b8ff57' : '#333333', transition: 'color 0.3s' }}>
                                    {fmtElapsed(elapsed)}
                                </span>
                            </div>

                            {/* Pipeline bar */}
                            <div>
                                <div style={{ display: 'flex', gap: '2px', height: '6px', marginBottom: '6px' }}>
                                    {segAgents.map((agent) => {
                                        const st = agents[agent].status
                                        const isRun = st === 'running'
                                        const isDone = st === 'done'
                                        return (
                                            <div key={agent} style={{
                                                flex: 1,
                                                borderRadius: '2px',
                                                background: isDone ? '#b8ff57' : isRun ? '#ff4d00' : '#1a1a1a',
                                                boxShadow: isRun ? '4px 0 12px rgba(255,77,0,0.6)' : 'none',
                                                transition: 'background 0.3s',
                                            }} />
                                        )
                                    })}
                                </div>
                                <div style={{ display: 'flex', gap: '2px' }}>
                                    {segLabels.map(lbl => (
                                        <div key={lbl} style={{ flex: 1, fontFamily: '"JetBrains Mono", monospace', fontSize: '9px', color: '#555555', textAlign: 'center' }}>
                                            {lbl}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Agent cards */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                                {AGENT_LIST.map(agent => (
                                    <HeroDitheringCard key={agent} style={cardStyle(agents[agent].status)}>
                                        <AgentCard name={agent} state={agents[agent]} />
                                    </HeroDitheringCard>
                                ))}
                            </div>
                        </div>
                    </WrapShader>

                </div>

                {/* ── BENCHMARK FOOTER ─────────────────────────────────────────── */}
                {benchmark && (
                    <div className="benchmark-footer" style={{
                        flexShrink: 0,
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: '24px',
                        padding: '16px 24px',
                        background: '#0c0c0c',
                        borderTop: '1px solid #1a1a1a',
                    }}>
                        {[
                            { label: 'CHANGES MADE', value: benchmark.total_changes, color: '#ffd60a' },
                            { label: 'BUGS FOUND', value: benchmark.bugs_found, color: '#ff3366' },
                            {
                                label: 'COMPILE STATUS',
                                value: benchmark.compiled_successfully ? 'SUCCESS' : 'FAILED',
                                color: benchmark.compiled_successfully ? '#b8ff57' : '#ff3366',
                            },
                            { label: 'DATA SOURCE', value: benchmark.data_source, color: '#b3e8ff', isSource: true },
                        ].map(({ label, value, color, isSource }) => (
                            <div key={label} style={{ display: 'flex', flexDirection: 'column', gap: '4px', minWidth: '120px' }}>
                                <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '10px', color: '#555555', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
                                    {label}
                                </span>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <AnimatedShinyText>
                                        <span style={{ fontFamily: '"JetBrains Mono", monospace', fontSize: '24px', fontWeight: 600, color, lineHeight: 1.1 }}>
                                            {String(value ?? '\u2014')}
                                        </span>
                                    </AnimatedShinyText>
                                    {isSource && value === 'real_rocm' && (
                                        <HeroBadge color="green">LIVE HARDWARE</HeroBadge>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

            </div>
        </>
    )
}
