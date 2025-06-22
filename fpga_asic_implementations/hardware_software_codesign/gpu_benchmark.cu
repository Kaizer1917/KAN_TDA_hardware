#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <vector>

struct BenchmarkResult {
    float execution_time_ms;
    float memory_bandwidth_gb_s;
    float throughput_gflops;
    int num_iterations;
    size_t memory_usage_bytes;
};

struct PerformanceCounters {
    cudaEvent_t start, stop;
    float elapsed_time;
    size_t peak_memory_usage;
    int active_warps;
    int occupancy_percentage;
};

__global__ void compute_intensive_kernel(const float* input, float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = input[idx];
        
        for (int i = 0; i < iterations; i++) {
            value = sinf(value) * cosf(value) + sqrtf(fabsf(value));
            value = value * 1.1f + 0.01f;
        }
        
        output[idx] = value;
    }
}

__global__ void memory_bandwidth_test_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    while (idx < n) {
        output[idx] = input[idx] * 2.0f + 1.0f;
        idx += stride;
    }
}

__global__ void latency_test_kernel(float* data, int n, int num_iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        for (int i = 0; i < num_iterations; i++) {
            value = __expf(value * 0.001f);
            value = __logf(value + 1.0f);
            value = __powf(value, 0.5f);
        }
        
        data[idx] = value;
    }
}

__global__ void shared_memory_test_kernel(const float* input, float* output, int n) {
    __shared__ float shared_data[1024];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && tid < 1024) {
        shared_data[tid] = input[idx];
    }
    
    __syncthreads();
    
    if (idx < n && tid < 1024) {
        float sum = 0.0f;
        for (int i = 0; i < min(blockDim.x, 1024); i++) {
            sum += shared_data[i];
        }
        output[idx] = sum / min(blockDim.x, 1024);
    }
}

__global__ void register_pressure_test_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float r0 = input[idx];
        float r1 = r0 * 1.1f;
        float r2 = r1 + 0.1f;
        float r3 = r2 * r0;
        float r4 = r3 - r1;
        float r5 = r4 + r2;
        float r6 = r5 * r3;
        float r7 = r6 - r4;
        float r8 = r7 + r5;
        float r9 = r8 * r6;
        float r10 = r9 - r7;
        float r11 = r10 + r8;
        float r12 = r11 * r9;
        float r13 = r12 - r10;
        float r14 = r13 + r11;
        float r15 = r14 * r12;
        
        output[idx] = r15 + r14 + r13 + r12 + r11 + r10 + r9 + r8;
    }
}

__global__ void warp_divergence_test_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = input[idx];
        
        if (idx % 32 < 16) {
            for (int i = 0; i < 100; i++) {
                value = sinf(value) + cosf(value);
            }
        } else {
            for (int i = 0; i < 50; i++) {
                value = sqrtf(fabsf(value)) + expf(value * 0.001f);
            }
        }
        
        output[idx] = value;
    }
}

__global__ void atomic_operations_test_kernel(float* data, int* counters, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        atomicAdd(&data[idx % 1024], 1.0f);
        atomicAdd(&counters[idx % 256], 1);
        
        if (idx % 32 == 0) {
            atomicMax(&counters[0], idx);
        }
    }
}

extern "C" {

BenchmarkResult benchmark_compute_throughput(int n, int num_iterations, int compute_intensity) {
    BenchmarkResult result = {0};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_input, n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        compute_intensive_kernel<<<grid_size, block_size>>>(d_input, d_output, n, compute_intensity);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    result.execution_time_ms = elapsed_time;
    result.num_iterations = num_iterations;
    result.throughput_gflops = (float)(n * compute_intensity * num_iterations * 10) / (elapsed_time * 1e6);
    result.memory_usage_bytes = 2 * n * sizeof(float);
    
    cudaFree(d_input);
    cudaFree(d_output);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult benchmark_memory_bandwidth(int n, int num_iterations) {
    BenchmarkResult result = {0};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_input, n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size(min(65535, (n + block_size.x - 1) / block_size.x));
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        memory_bandwidth_test_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    size_t bytes_transferred = 2 * n * sizeof(float) * num_iterations;
    result.execution_time_ms = elapsed_time;
    result.memory_bandwidth_gb_s = (float)bytes_transferred / (elapsed_time * 1e6);
    result.num_iterations = num_iterations;
    result.memory_usage_bytes = 2 * n * sizeof(float);
    
    cudaFree(d_input);
    cudaFree(d_output);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult benchmark_latency(int n, int num_iterations, int math_ops_per_thread) {
    BenchmarkResult result = {0};
    
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_data, n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        latency_test_kernel<<<grid_size, block_size>>>(d_data, n, math_ops_per_thread);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    result.execution_time_ms = elapsed_time;
    result.num_iterations = num_iterations;
    result.memory_usage_bytes = n * sizeof(float);
    
    cudaFree(d_data);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult benchmark_shared_memory(int n, int num_iterations) {
    BenchmarkResult result = {0};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_input, n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        shared_memory_test_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    result.execution_time_ms = elapsed_time;
    result.num_iterations = num_iterations;
    result.memory_usage_bytes = 2 * n * sizeof(float);
    
    cudaFree(d_input);
    cudaFree(d_output);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult benchmark_register_pressure(int n, int num_iterations) {
    BenchmarkResult result = {0};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_input, n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        register_pressure_test_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    result.execution_time_ms = elapsed_time;
    result.num_iterations = num_iterations;
    result.memory_usage_bytes = 2 * n * sizeof(float);
    
    cudaFree(d_input);
    cudaFree(d_output);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult benchmark_warp_divergence(int n, int num_iterations) {
    BenchmarkResult result = {0};
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_input, n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        warp_divergence_test_kernel<<<grid_size, block_size>>>(d_input, d_output, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    result.execution_time_ms = elapsed_time;
    result.num_iterations = num_iterations;
    result.memory_usage_bytes = 2 * n * sizeof(float);
    
    cudaFree(d_input);
    cudaFree(d_output);
    curandDestroyGenerator(gen);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

BenchmarkResult benchmark_atomic_operations(int n, int num_iterations) {
    BenchmarkResult result = {0};
    
    float *d_data;
    int *d_counters;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    cudaMalloc(&d_counters, 256 * sizeof(int));
    
    cudaMemset(d_data, 0, 1024 * sizeof(float));
    cudaMemset(d_counters, 0, 256 * sizeof(int));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    cudaEventRecord(start);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        atomic_operations_test_kernel<<<grid_size, block_size>>>(d_data, d_counters, n);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    result.execution_time_ms = elapsed_time;
    result.num_iterations = num_iterations;
    result.memory_usage_bytes = 1024 * sizeof(float) + 256 * sizeof(int);
    
    cudaFree(d_data);
    cudaFree(d_counters);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

void run_comprehensive_benchmark() {
    int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("=== GPU Performance Benchmark Results ===\n\n");
    
    printf("Compute Throughput Benchmark:\n");
    printf("Size\t\tTime(ms)\tGFLOPS\t\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_compute_throughput(sizes[i], 100, 50);
        printf("%d\t\t%.2f\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, result.throughput_gflops, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
    
    printf("\nMemory Bandwidth Benchmark:\n");
    printf("Size\t\tTime(ms)\tBW(GB/s)\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_memory_bandwidth(sizes[i], 100);
        printf("%d\t\t%.2f\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, result.memory_bandwidth_gb_s, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
    
    printf("\nLatency Benchmark:\n");
    printf("Size\t\tTime(ms)\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_latency(sizes[i], 10, 100);
        printf("%d\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
    
    printf("\nShared Memory Benchmark:\n");
    printf("Size\t\tTime(ms)\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_shared_memory(sizes[i], 100);
        printf("%d\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
    
    printf("\nRegister Pressure Benchmark:\n");
    printf("Size\t\tTime(ms)\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_register_pressure(sizes[i], 100);
        printf("%d\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
    
    printf("\nWarp Divergence Benchmark:\n");
    printf("Size\t\tTime(ms)\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_warp_divergence(sizes[i], 50);
        printf("%d\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
    
    printf("\nAtomic Operations Benchmark:\n");
    printf("Size\t\tTime(ms)\tMemory(MB)\n");
    for (int i = 0; i < num_sizes; i++) {
        BenchmarkResult result = benchmark_atomic_operations(sizes[i], 100);
        printf("%d\t\t%.2f\t\t%.2f\n", 
               sizes[i], result.execution_time_ms, 
               result.memory_usage_bytes / (1024.0f * 1024.0f));
    }
}

} 