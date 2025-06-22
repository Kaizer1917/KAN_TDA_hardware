#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

#define WARP_SIZE 32
#define CACHE_LINE_SIZE 128
#define SHARED_MEMORY_SIZE 49152

struct SparseMatrix {
    int* row_ptr;
    int* col_idx;
    float* values;
    int num_rows;
    int num_cols;
    int nnz;
};

struct MemoryPool {
    void* base_ptr;
    size_t total_size;
    size_t* block_sizes;
    bool* block_free;
    int num_blocks;
    int current_block;
};

__device__ __forceinline__ int get_cache_aligned_size(int size) {
    return ((size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
}

__global__ void sparse_matrix_construction_kernel(const float* dense_matrix, 
                                                 int* row_ptr, int* col_idx, float* values,
                                                 int rows, int cols, float threshold) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        int nnz_count = 0;
        
        for (int col = 0; col < cols; col++) {
            float val = dense_matrix[row * cols + col];
            if (fabsf(val) > threshold) {
                nnz_count++;
            }
        }
        
        row_ptr[row + 1] = nnz_count;
    }
    
    __syncthreads();
    
    if (row == 0) {
        row_ptr[0] = 0;
        for (int i = 1; i <= rows; i++) {
            row_ptr[i] += row_ptr[i - 1];
        }
    }
    
    __syncthreads();
    
    if (row < rows) {
        int start_idx = row_ptr[row];
        int current_idx = start_idx;
        
        for (int col = 0; col < cols; col++) {
            float val = dense_matrix[row * cols + col];
            if (fabsf(val) > threshold) {
                col_idx[current_idx] = col;
                values[current_idx] = val;
                current_idx++;
            }
        }
    }
}

__global__ void memory_coalesced_matrix_multiply(const float* A, const float* B, float* C,
                                                int M, int N, int K) {
    __shared__ float As[32][33];
    __shared__ float Bs[32][33];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (K + 31) / 32; k++) {
        if (row < M && (k * 32 + tx) < K) {
            As[ty][tx] = A[row * K + k * 32 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((k * 32 + ty) < K && col < N) {
            Bs[ty][tx] = B[(k * 32 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < 32; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void streaming_data_processing_kernel(const float* input_stream, float* output_stream,
                                                float* buffer1, float* buffer2, 
                                                int chunk_size, int iteration) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < chunk_size) {
        float* current_buffer = (iteration % 2 == 0) ? buffer1 : buffer2;
        float* previous_buffer = (iteration % 2 == 0) ? buffer2 : buffer1;
        
        current_buffer[idx] = input_stream[idx] * 2.0f;
        
        if (iteration > 0) {
            output_stream[idx] = current_buffer[idx] + previous_buffer[idx] * 0.5f;
        } else {
            output_stream[idx] = current_buffer[idx];
        }
    }
}

__global__ void cache_efficient_transpose_kernel(const float* input, float* output,
                                                int rows, int cols) {
    __shared__ float tile[32][33];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void memory_pool_allocation_kernel(void* memory_pool, size_t* block_sizes,
                                             bool* block_free, void** allocated_ptrs,
                                             size_t* requested_sizes, int num_requests) {
    int request_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (request_idx < num_requests) {
        size_t required_size = requested_sizes[request_idx];
        allocated_ptrs[request_idx] = nullptr;
        
        for (int block = 0; block < num_requests; block++) {
            if (block_free[block] && block_sizes[block] >= required_size) {
                if (atomicCAS((int*)&block_free[block], 1, 0) == 1) {
                    allocated_ptrs[request_idx] = (char*)memory_pool + block * 1024 * 1024;
                    break;
                }
            }
        }
    }
}

__global__ void hierarchical_memory_access_kernel(float* L1_cache, float* L2_cache, 
                                                 float* global_memory, 
                                                 int* access_pattern, int pattern_size) {
    __shared__ float shared_cache[1024];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    if (tid < 1024 && idx < pattern_size) {
        int global_idx = access_pattern[idx];
        shared_cache[tid] = global_memory[global_idx];
    }
    
    __syncthreads();
    
    if (idx < pattern_size) {
        float result = shared_cache[tid] * 1.5f + shared_cache[(tid + 1) % 1024] * 0.5f;
        global_memory[access_pattern[idx]] = result;
    }
}

__global__ void compressed_storage_kernel(const float* input_data, int* compressed_indices,
                                         float* compressed_values, int* count,
                                         int data_size, float compression_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < data_size) {
        float value = input_data[idx];
        if (fabsf(value) > compression_threshold) {
            int compressed_idx = atomicAdd(count, 1);
            compressed_indices[compressed_idx] = idx;
            compressed_values[compressed_idx] = value;
        }
    }
}

__global__ void vectorized_reduction_kernel(const float4* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (i < n) {
        float4 val = input[i];
        sum = val.x + val.y + val.z + val.w;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void bandwidth_optimized_copy_kernel(const float* src, float* dst, 
                                               int size, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    
    if (idx + stride <= size) {
        float4* src4 = (float4*)&src[idx];
        float4* dst4 = (float4*)&dst[idx];
        
        for (int i = 0; i < stride / 4; i++) {
            dst4[i] = src4[i];
        }
    }
}

__global__ void adaptive_memory_layout_kernel(const float* input_matrix, float* aos_output,
                                             float* soa_output, int rows, int cols,
                                             bool use_aos_layout) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        float value = input_matrix[row * cols + col];
        
        if (use_aos_layout) {
            aos_output[row * cols + col] = value;
        } else {
            soa_output[col * rows + row] = value;
        }
    }
}

extern "C" {

void initialize_memory_pool(MemoryPool* pool, size_t total_size, int num_blocks) {
    cudaMalloc(&pool->base_ptr, total_size);
    pool->total_size = total_size;
    pool->num_blocks = num_blocks;
    pool->current_block = 0;
    
    cudaMallocManaged(&pool->block_sizes, num_blocks * sizeof(size_t));
    cudaMallocManaged(&pool->block_free, num_blocks * sizeof(bool));
    
    size_t block_size = total_size / num_blocks;
    for (int i = 0; i < num_blocks; i++) {
        pool->block_sizes[i] = block_size;
        pool->block_free[i] = true;
    }
}

void sparse_matrix_conversion(const float* dense_matrix, SparseMatrix* sparse,
                             int rows, int cols, float threshold) {
    
    int* d_row_ptr;
    int* d_col_idx_temp;
    float* d_values_temp;
    int* d_nnz_count;
    
    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx_temp, rows * cols * sizeof(int));
    cudaMalloc(&d_values_temp, rows * cols * sizeof(float));
    cudaMalloc(&d_nnz_count, sizeof(int));
    
    cudaMemset(d_nnz_count, 0, sizeof(int));
    
    dim3 block_size(256);
    dim3 grid_size((rows + block_size.x - 1) / block_size.x);
    
    sparse_matrix_construction_kernel<<<grid_size, block_size>>>(
        dense_matrix, d_row_ptr, d_col_idx_temp, d_values_temp, rows, cols, threshold);
    
    int h_nnz_count;
    cudaMemcpy(&h_nnz_count, &d_row_ptr[rows], sizeof(int), cudaMemcpyDeviceToHost);
    
    sparse->num_rows = rows;
    sparse->num_cols = cols;
    sparse->nnz = h_nnz_count;
    
    cudaMalloc(&sparse->row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&sparse->col_idx, h_nnz_count * sizeof(int));
    cudaMalloc(&sparse->values, h_nnz_count * sizeof(float));
    
    cudaMemcpy(sparse->row_ptr, d_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sparse->col_idx, d_col_idx_temp, h_nnz_count * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(sparse->values, d_values_temp, h_nnz_count * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx_temp);
    cudaFree(d_values_temp);
    cudaFree(d_nnz_count);
}

void streaming_computation(float* input_data, float* output_data, 
                          int total_size, int chunk_size) {
    
    float* d_buffer1;
    float* d_buffer2;
    float* d_input_chunk;
    float* d_output_chunk;
    
    cudaMalloc(&d_buffer1, chunk_size * sizeof(float));
    cudaMalloc(&d_buffer2, chunk_size * sizeof(float));
    cudaMalloc(&d_input_chunk, chunk_size * sizeof(float));
    cudaMalloc(&d_output_chunk, chunk_size * sizeof(float));
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    int num_chunks = (total_size + chunk_size - 1) / chunk_size;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int current_chunk_size = min(chunk_size, total_size - chunk * chunk_size);
        
        cudaMemcpyAsync(d_input_chunk, &input_data[chunk * chunk_size],
                       current_chunk_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
        
        dim3 block_size(256);
        dim3 grid_size((current_chunk_size + block_size.x - 1) / block_size.x);
        
        streaming_data_processing_kernel<<<grid_size, block_size, 0, stream1>>>(
            d_input_chunk, d_output_chunk, d_buffer1, d_buffer2, current_chunk_size, chunk);
        
        cudaMemcpyAsync(&output_data[chunk * chunk_size], d_output_chunk,
                       current_chunk_size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    }
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaFree(d_buffer1);
    cudaFree(d_buffer2);
    cudaFree(d_input_chunk);
    cudaFree(d_output_chunk);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

void cache_optimized_matrix_operations(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    
    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, 
                   (M + block_size.y - 1) / block_size.y);
    
    memory_coalesced_matrix_multiply<<<grid_size, block_size>>>(A, B, C, M, N, K);
    
    cudaDeviceSynchronize();
}

void hierarchical_memory_management(float* data, int data_size, 
                                   int* access_pattern, int pattern_size) {
    
    float* d_L1_cache;
    float* d_L2_cache;
    int* d_access_pattern;
    
    cudaMalloc(&d_L1_cache, 32 * 1024);
    cudaMalloc(&d_L2_cache, 512 * 1024);
    cudaMalloc(&d_access_pattern, pattern_size * sizeof(int));
    
    cudaMemcpy(d_access_pattern, access_pattern, pattern_size * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 block_size(256);
    dim3 grid_size((pattern_size + block_size.x - 1) / block_size.x);
    
    hierarchical_memory_access_kernel<<<grid_size, block_size>>>(
        d_L1_cache, d_L2_cache, data, d_access_pattern, pattern_size);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_L1_cache);
    cudaFree(d_L2_cache);
    cudaFree(d_access_pattern);
}

void adaptive_compression(const float* input_data, int** compressed_indices,
                         float** compressed_values, int* compressed_size,
                         int data_size, float threshold) {
    
    int* d_compressed_indices;
    float* d_compressed_values;
    int* d_count;
    
    cudaMalloc(&d_compressed_indices, data_size * sizeof(int));
    cudaMalloc(&d_compressed_values, data_size * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    dim3 block_size(256);
    dim3 grid_size((data_size + block_size.x - 1) / block_size.x);
    
    compressed_storage_kernel<<<grid_size, block_size>>>(
        input_data, d_compressed_indices, d_compressed_values, d_count, data_size, threshold);
    
    cudaMemcpy(compressed_size, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    *compressed_indices = (int*)malloc(*compressed_size * sizeof(int));
    *compressed_values = (float*)malloc(*compressed_size * sizeof(float));
    
    cudaMemcpy(*compressed_indices, d_compressed_indices, *compressed_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*compressed_values, d_compressed_values, *compressed_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_compressed_indices);
    cudaFree(d_compressed_values);
    cudaFree(d_count);
}

} 