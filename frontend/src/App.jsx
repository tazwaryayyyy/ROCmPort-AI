import { useState, useEffect, useRef } from 'react'

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

// Tailwind class strings per status — all literals so JIT can scan them
const STATUS = {
    idle: {
        dot: 'bg-[#1E2D40]',
        badge: 'bg-[#1E2D40] text-[#6B7A99]',
        label: 'IDLE',
    },
    running: {
        dot: 'bg-[#FFB800] animate-rocm-pulse',
        badge: 'bg-[#1A1500] text-[#FFB800]',
        label: 'RUNNING',
    },
    done: {
        dot: 'bg-[#00FF88]',
        badge: 'bg-[#001A0D] text-[#00FF88]',
        label: 'DONE',
    },
    failed: {
        dot: 'bg-[#FF3B3B]',
        badge: 'bg-[#1A0000] text-[#FF3B3B]',
        label: 'FAILED',
    },
}

const INITIAL_AGENTS = Object.fromEntries(
    AGENT_LIST.map(a => [a, { status: 'idle', message: 'Waiting…', detail: '' }])
)

// ─── AgentCard ────────────────────────────────────────────────────────────────

function AgentCard({ name, state }) {
    const s = STATUS[state.status] ?? STATUS.idle
    return (
        <div className="rounded-lg border border-[#1E2D40] bg-[#111827] p-3">
            <div className="flex items-center gap-3">
                {/* Status dot */}
                <span className={`shrink-0 w-2 h-2 rounded-full ${s.dot}`} />

                {/* Agent info */}
                <div className="flex-1 min-w-0">
                    <div className="font-code text-[11px] text-[#6B7A99] tracking-widest uppercase">
                        {AGENT_LABEL[name]}
                    </div>
                    <div className="font-ui text-[13px] text-[#F0F4FF] mt-0.5 truncate">
                        {state.message || 'Waiting…'}
                    </div>
                </div>

                {/* Status badge */}
                <span
                    className={`shrink-0 font-code text-[10px] font-semibold px-2 py-0.5 rounded tracking-wider ${s.badge}`}
                >
                    {s.label}
                </span>
            </div>

            {/* Detail (collapsible — shown only when present) */}
            {state.detail && (
                <p className="mt-2 font-ui text-[11px] text-[#6B7A99] italic leading-relaxed line-clamp-3">
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

    const lineCount = code ? code.split('\n').length : 1

    // ── Timer ────────────────────────────────────────────────────────────────────
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

    // ── Helpers ───────────────────────────────────────────────────────────────────
    const resetAgents = () =>
        setAgents(Object.fromEntries(
            AGENT_LIST.map(a => [a, { status: 'idle', message: 'Waiting…', detail: '' }])
        ))

    const updateAgent = (agent, patch) =>
        setAgents(prev => ({ ...prev, [agent]: { ...prev[agent], ...patch } }))

    const selectTemplate = (name) => {
        setActiveTemplate(name)
        setCode(TEMPLATES[name])
    }

    const fmtElapsed = (ms) => `${(ms / 1000).toFixed(1)}s`

    // ── Demo mode fallback ────────────────────────────────────────────────────────
    const runDemo = async () => {
        const steps = [
            { agent: 'analyzer', status: 'running', message: 'Scanning CUDA patterns…', detail: '' },
            { agent: 'analyzer', status: 'done', message: 'Found 3 critical AMD issues', detail: 'warp-32 assumption in reduction tail, threadIdx%32 idiom, LDS bank conflict pattern' },
            { agent: 'translator', status: 'running', message: 'Running hipify + LLM pass…', detail: '' },
            { agent: 'translator', status: 'done', message: 'Translation complete', detail: 'hipify applied; 7 additional LLM corrections for wavefront-64 semantics' },
            { agent: 'optimizer', status: 'running', message: 'Proposing optimizations…', detail: '' },
            { agent: 'optimizer', status: 'done', message: '4 optimization patches generated', detail: 'LDS padding, wavefront-aware reduction, coalesced access pattern' },
            { agent: 'tester', status: 'running', message: 'Compiling with hipcc…', detail: '' },
            { agent: 'tester', status: 'done', message: 'Compiled and profiled on gfx942', detail: 'rocprof: 0.026 ms — correctness verified' },
            { agent: 'coordinator', status: 'running', message: 'Assembling final report…', detail: '' },
            { agent: 'coordinator', status: 'done', message: 'Migration complete — 2.61× speedup', detail: 'data_source: demo_artifact' },
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

    // ── Main action ───────────────────────────────────────────────────────────────
    const handlePort = async () => {
        if (running || !code.trim()) return

        setRunning(true)
        setElapsed(0)
        setBenchmark(null)
        setErrorBanner(null)
        resetAgents()
        startTimer()

        try {
            const res = await fetch('http://localhost:8000/port', {
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
                buf = lines.pop() // keep any incomplete trailing line

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

                        // Extract benchmark data from the coordinator's done event
                        if (ev.agent === 'coordinator' && ev.status === 'done') {
                            let report = ev.result ?? null
                            if (!report && ev.detail) {
                                try {
                                    report = JSON.parse(ev.detail)
                                } catch (_) {
                                    report = null
                                }
                            }
                            const r = report ?? ev
                            setBenchmark({
                                total_changes: r.total_changes ?? r.changes_made ?? '—',
                                bugs_found: r.bugs_found ?? r.critical_bugs ?? r.static_risk_report?.critical_count ?? '—',
                                compiled_successfully: r.compiled_successfully ?? r.compiled ?? r.migration_success ?? false,
                                data_source: r.data_source ?? 'unknown',
                            })
                        }
                    } catch (_) { /* malformed SSE line — skip */ }
                }
            }
        } catch {
            setErrorBanner('Backend unavailable — running in demo mode')
            runDemo()
            return // runDemo handles stopTimer + setRunning(false)
        }

        stopTimer()
        setRunning(false)
    }

    // ── Render ────────────────────────────────────────────────────────────────────
    return (
        <div
            className="min-h-screen flex flex-col text-[#F0F4FF] font-ui"
            style={{ background: 'linear-gradient(180deg, #0A0E1A 0%, #0D1220 100%)' }}
        >
            {/* ── Error banner ──────────────────────────────────────────────────────── */}
            {errorBanner && (
                <div className="flex-none px-6 py-2.5 border-b border-[#FF3B3B] bg-[#1A0000] font-code text-[13px] text-[#FF3B3B]">
                    ⚠ {errorBanner}
                </div>
            )}

            {/* ── Two-column main layout ────────────────────────────────────────────── */}
            <div className="flex flex-1 overflow-hidden">

                {/* ──── LEFT PANEL  58% ─────────────────────────────────────────────── */}
                <div className="w-[58%] flex flex-col p-5 gap-4 border-r border-[#1E2D40] overflow-y-auto">

                    {/* Editor header */}
                    <div className="flex justify-between items-center">
                        <span className="font-code text-[12px] text-[#6B7A99]">// CUDA source</span>
                        <span className="font-code text-[12px] text-[#6B7A99]">{lineCount} lines</span>
                    </div>

                    {/* Code editor */}
                    <textarea
                        value={code}
                        onChange={e => { setCode(e.target.value); setActiveTemplate(null) }}
                        placeholder={'// Paste CUDA code here\n// or pick a demo below'}
                        spellCheck={false}
                        className={[
                            'w-full min-h-[300px] resize-y rounded-lg p-4',
                            'border border-[#1E2D40] bg-[#0D1525]',
                            'text-[#F0F4FF] font-code text-[13px] leading-[1.6]',
                            'focus:outline-none focus:border-[#00D4FF] transition-colors duration-150',
                            '[tab-size:4] [caret-color:#00D4FF]',
                        ].join(' ')}
                    />

                    {/* Template selector */}
                    <div>
                        <p className="font-ui text-[12px] text-[#6B7A99] mb-2.5">Select a template:</p>
                        <div className="flex flex-wrap gap-2">
                            {Object.keys(TEMPLATES).map(name => (
                                <button
                                    key={name}
                                    onClick={() => selectTemplate(name)}
                                    className={[
                                        'px-4 py-1.5 rounded-full border font-ui text-[13px]',
                                        'cursor-pointer transition-colors duration-150',
                                        activeTemplate === name
                                            ? 'bg-[#001A24] border-[#00D4FF] text-[#00D4FF]'
                                            : 'bg-[#111827] border-[#1E2D40] text-[#F0F4FF] hover:border-[#00D4FF]',
                                    ].join(' ')}
                                >
                                    {name}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* PORT TO ROCM button */}
                    <button
                        onClick={handlePort}
                        disabled={running || !code.trim()}
                        className={[
                            'w-full h-12 rounded-lg font-code text-[14px] text-white font-semibold',
                            '[letter-spacing:2px] transition-all duration-150',
                            running || !code.trim()
                                ? 'bg-[#FF3B3B] opacity-50 cursor-not-allowed'
                                : 'bg-[#FF3B3B] hover:bg-[#FF1A1A] hover:shadow-[0_0_20px_rgba(255,59,59,0.35)] cursor-pointer',
                        ].join(' ')}
                    >
                        {running ? 'RUNNING...' : 'PORT TO ROCM'}
                    </button>
                </div>

                {/* ──── RIGHT PANEL  42% ────────────────────────────────────────────── */}
                <div className="w-[42%] flex flex-col p-5 gap-4 overflow-y-auto">

                    {/* Pipeline header */}
                    <div className="flex justify-between items-center">
                        <span className="font-code text-[12px] text-[#6B7A99]">// Pipeline</span>
                        <span className={`font-code text-[12px] transition-colors duration-300 ${running ? 'text-[#FFB800]' : 'text-[#6B7A99]'}`}>
                            {fmtElapsed(elapsed)}
                        </span>
                    </div>

                    {/* Agent cards */}
                    <div className="flex flex-col gap-2">
                        {AGENT_LIST.map(agent => (
                            <AgentCard key={agent} name={agent} state={agents[agent]} />
                        ))}
                    </div>
                </div>
            </div>

            {/* ── Benchmark footer (hidden until run completes) ─────────────────────── */}
            {benchmark && (
                <div className="flex-none flex flex-wrap gap-6 px-6 py-4 border-t border-[#1E2D40] bg-[#0D1525]">
                    {[
                        { label: 'CHANGES MADE', value: benchmark.total_changes },
                        { label: 'BUGS FOUND', value: benchmark.bugs_found },
                        {
                            label: 'COMPILE STATUS',
                            value: benchmark.compiled_successfully ? 'SUCCESS' : 'FAILED',
                            color: benchmark.compiled_successfully ? '#00FF88' : '#FF3B3B',
                        },
                        { label: 'DATA SOURCE', value: benchmark.data_source, isSource: true },
                    ].map(({ label, value, color, isSource }) => (
                        <div key={label} className="flex flex-col gap-1 min-w-[120px]">
                            <span className="font-ui text-[10px] text-[#6B7A99] uppercase tracking-widest">
                                {label}
                            </span>
                            <div className="flex items-center gap-2">
                                <span
                                    className="font-code text-[18px] font-semibold"
                                    style={{ color: color ?? '#00D4FF' }}
                                >
                                    {String(value ?? '—')}
                                </span>
                                {isSource && value === 'real_rocm' && (
                                    <span className="font-code text-[10px] text-[#00D4FF] border border-[#00D4FF] bg-[#001A24] px-2 py-0.5 rounded">
                                        LIVE HARDWARE
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
