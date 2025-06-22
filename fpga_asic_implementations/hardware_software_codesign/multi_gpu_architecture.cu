#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <memory>

struct GPUContext {
    int device_id;
    cudaStream_t compute_stream;
    cudaStream_t comm_stream;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    ncclComm_t nccl_comm;
    float* device_memory_pool;
    size_t memory_pool_size;
    size_t memory_offset;
};

class MultiGPUTDAManager {
private:
    std::vector<GPUContext> gpu_contexts;
    int num_gpus;
    int rank;
    int world_size;
    
public:
    MultiGPUTDAManager(int num_devices) : num_gpus(num_devices) {
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        gpu_contexts.resize(num_gpus);
        ncclUniqueId nccl_id;
        
        if (rank == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        for (int i = 0; i < num_gpus; i++) {
            gpu_contexts[i].device_id = rank * num_gpus + i;
            cudaSetDevice(gpu_contexts[i].device_id);
            
            cudaStreamCreate(&gpu_contexts[i].compute_stream);
            cudaStreamCreate(&gpu_contexts[i].comm_stream);
            cublasCreate(&gpu_contexts[i].cublas_handle);
            cusparseCreate(&gpu_contexts[i].cusparse_handle);
            
            cublasSetStream(gpu_contexts[i].cublas_handle, gpu_contexts[i].compute_stream);
            cusparseSetStream(gpu_contexts[i].cusparse_handle, gpu_contexts[i].compute_stream);
            
            ncclCommInitRank(&gpu_contexts[i].nccl_comm, world_size * num_gpus, 
                           nccl_id, rank * num_gpus + i);
            
            gpu_contexts[i].memory_pool_size = 8ULL * 1024 * 1024 * 1024;
            cudaMalloc(&gpu_contexts[i].device_memory_pool, gpu_contexts[i].memory_pool_size);
            gpu_contexts[i].memory_offset = 0;
        }
    }
    
    ~MultiGPUTDAManager() {
        for (auto& ctx : gpu_contexts) {
            cudaSetDevice(ctx.device_id);
            cudaStreamDestroy(ctx.compute_stream);
            cudaStreamDestroy(ctx.comm_stream);
            cublasDestroy(ctx.cublas_handle);
            cusparseDestroy(ctx.cusparse_handle);
            ncclCommDestroy(ctx.nccl_comm);
            cudaFree(ctx.device_memory_pool);
        }
        MPI_Finalize();
    }
    
    void* allocate_gpu_memory(int gpu_id, size_t size) {
        auto& ctx = gpu_contexts[gpu_id];
        if (ctx.memory_offset + size > ctx.memory_pool_size) {
            return nullptr;
        }
        void* ptr = (char*)ctx.device_memory_pool + ctx.memory_offset;
        ctx.memory_offset += size;
        return ptr;
    }
    
    void reset_memory_pool(int gpu_id) {
        gpu_contexts[gpu_id].memory_offset = 0;
    }
};

__global__ void hierarchical_decomposition_kernel(const float* input_data, float** output_chunks,
                                                 int* chunk_sizes, int total_size, int num_chunks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_id = blockIdx.y;
    
    if (chunk_id < num_chunks && idx < chunk_sizes[chunk_id]) {
        int offset = 0;
        for (int i = 0; i < chunk_id; i++) {
            offset += chunk_sizes[i];
        }
        output_chunks[chunk_id][idx] = input_data[offset + idx];
    }
}

__global__ void load_balancing_kernel(const int* workload_sizes, int* gpu_assignments,
                                    float* gpu_loads, int num_tasks, int num_gpus) {
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (task_id < num_tasks) {
        float min_load = gpu_loads[0];
        int best_gpu = 0;
        
        for (int gpu = 1; gpu < num_gpus; gpu++) {
            if (gpu_loads[gpu] < min_load) {
                min_load = gpu_loads[gpu];
                best_gpu = gpu;
            }
        }
        
        gpu_assignments[task_id] = best_gpu;
        atomicAdd(&gpu_loads[best_gpu], (float)workload_sizes[task_id]);
    }
}

__global__ void distributed_distance_computation_kernel(const float* points_local, 
                                                       const float* points_remote,
                                                       float* distances, 
                                                       int n_local, int n_remote, int dim,
                                                       int local_offset, int remote_offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_local && j < n_remote) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = points_local[i * dim + d] - points_remote[j * dim + d];
            sum += diff * diff;
        }
        distances[(local_offset + i) * (n_local + n_remote) + (remote_offset + j)] = sqrtf(sum);
    }
}

__global__ void communication_overlap_kernel(float* send_buffer, float* recv_buffer,
                                            int* send_counts, int* recv_counts,
                                            int num_neighbors) {
    int neighbor_id = blockIdx.x;
    int idx = threadIdx.x;
    
    if (neighbor_id < num_neighbors && idx < send_counts[neighbor_id]) {
        recv_buffer[neighbor_id * blockDim.x + idx] = 
            send_buffer[neighbor_id * blockDim.x + idx] * 2.0f;
    }
}

__global__ void parallel_merge_kernel(float** input_arrays, int* array_sizes,
                                     float* output_array, int num_arrays) {
    extern __shared__ int indices[];
    
    int tid = threadIdx.x;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_arrays) {
        indices[tid] = 0;
    }
    __syncthreads();
    
    if (output_idx == 0) {
        int total_elements = 0;
        for (int i = 0; i < num_arrays; i++) {
            total_elements += array_sizes[i];
        }
        
        for (int out_pos = 0; out_pos < total_elements; out_pos++) {
            float min_val = FLT_MAX;
            int min_array = -1;
            
            for (int arr = 0; arr < num_arrays; arr++) {
                if (indices[arr] < array_sizes[arr]) {
                    if (input_arrays[arr][indices[arr]] < min_val) {
                        min_val = input_arrays[arr][indices[arr]];
                        min_array = arr;
                    }
                }
            }
            
            if (min_array >= 0) {
                output_array[out_pos] = min_val;
                indices[min_array]++;
            }
        }
    }
}

__global__ void nccl_allreduce_preparation_kernel(float* local_data, float* global_buffer,
                                                 int local_size, int global_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < local_size) {
        global_buffer[global_offset + idx] = local_data[idx];
    }
}

__global__ void streaming_computation_kernel(const float* input_stream, float* output_stream,
                                           float* computation_buffer, int stream_size,
                                           int buffer_size, int iteration) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int buffer_offset = (iteration % 2) * buffer_size;
    
    if (idx < stream_size) {
        computation_buffer[buffer_offset + idx] = input_stream[idx] * 1.5f + 
                                                 computation_buffer[buffer_offset + idx] * 0.8f;
        
        if (iteration > 0) {
            output_stream[idx] = computation_buffer[buffer_offset + idx] + 
                               computation_buffer[(1 - iteration % 2) * buffer_size + idx];
        }
    }
}

extern "C" {

void multi_gpu_persistent_homology(float* h_points, int total_points, int dim,
                                  float epsilon, float* h_results, int num_gpus) {
    
    MultiGPUTDAManager manager(num_gpus);
    
    int points_per_gpu = total_points / (manager.world_size * num_gpus);
    int remainder = total_points % (manager.world_size * num_gpus);
    
    #pragma omp parallel for num_threads(num_gpus)
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(manager.gpu_contexts[gpu].device_id);
        
        int local_points = points_per_gpu;
        if (manager.rank * num_gpus + gpu < remainder) {
            local_points++;
        }
        
        int offset = (manager.rank * num_gpus + gpu) * points_per_gpu + 
                    min(manager.rank * num_gpus + gpu, remainder);
        
        float* d_local_points = (float*)manager.allocate_gpu_memory(gpu, 
                                                                   local_points * dim * sizeof(float));
        float* d_distances = (float*)manager.allocate_gpu_memory(gpu, 
                                                                local_points * local_points * sizeof(float));
        
        cudaMemcpyAsync(d_local_points, &h_points[offset * dim], 
                       local_points * dim * sizeof(float), cudaMemcpyHostToDevice,
                       manager.gpu_contexts[gpu].compute_stream);
        
        dim3 block_size(16, 16);
        dim3 grid_size((local_points + block_size.x - 1) / block_size.x,
                      (local_points + block_size.y - 1) / block_size.y);
        
        distributed_distance_computation_kernel<<<grid_size, block_size, 0, 
                                                manager.gpu_contexts[gpu].compute_stream>>>(
            d_local_points, d_local_points, d_distances, 
            local_points, local_points, dim, 0, 0);
        
        float* h_local_distances = new float[local_points * local_points];
        cudaMemcpyAsync(h_local_distances, d_distances, 
                       local_points * local_points * sizeof(float),
                       cudaMemcpyDeviceToHost, manager.gpu_contexts[gpu].compute_stream);
        
        cudaStreamSynchronize(manager.gpu_contexts[gpu].compute_stream);
        
        ncclAllReduce(d_distances, d_distances, local_points * local_points, ncclFloat, ncclSum,
                     manager.gpu_contexts[gpu].nccl_comm, manager.gpu_contexts[gpu].comm_stream);
        
        cudaStreamSynchronize(manager.gpu_contexts[gpu].comm_stream);
        
        manager.reset_memory_pool(gpu);
        delete[] h_local_distances;
    }
}

void hierarchical_sequence_decomposition(float* sequence, int sequence_length,
                                        int num_levels, float** decomposed_sequences,
                                        int* level_sizes) {
    
    for (int level = 0; level < num_levels; level++) {
        int decimation_factor = 1 << level;
        level_sizes[level] = sequence_length / decimation_factor;
        
        cudaMalloc(&decomposed_sequences[level], level_sizes[level] * sizeof(float));
        
        dim3 block_size(256);
        dim3 grid_size((level_sizes[level] + block_size.x - 1) / block_size.x);
        
        hierarchical_decomposition_kernel<<<grid_size, block_size>>>(
            sequence, &decomposed_sequences[level], &level_sizes[level], 
            sequence_length, 1);
        
        cudaDeviceSynchronize();
    }
}

void load_balanced_task_distribution(int* task_sizes, int num_tasks, 
                                    int num_gpus, int* gpu_assignments) {
    
    float* d_gpu_loads;
    int* d_task_sizes;
    int* d_gpu_assignments;
    
    cudaMalloc(&d_gpu_loads, num_gpus * sizeof(float));
    cudaMalloc(&d_task_sizes, num_tasks * sizeof(int));
    cudaMalloc(&d_gpu_assignments, num_tasks * sizeof(int));
    
    cudaMemset(d_gpu_loads, 0, num_gpus * sizeof(float));
    cudaMemcpy(d_task_sizes, task_sizes, num_tasks * sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 block_size(256);
    dim3 grid_size((num_tasks + block_size.x - 1) / block_size.x);
    
    load_balancing_kernel<<<grid_size, block_size>>>(d_task_sizes, d_gpu_assignments,
                                                    d_gpu_loads, num_tasks, num_gpus);
    
    cudaMemcpy(gpu_assignments, d_gpu_assignments, num_tasks * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_gpu_loads);
    cudaFree(d_task_sizes);
    cudaFree(d_gpu_assignments);
}

} 