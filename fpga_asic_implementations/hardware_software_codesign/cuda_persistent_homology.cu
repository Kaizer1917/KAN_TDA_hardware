#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

__device__ __forceinline__ float euclidean_distance_gpu(const float* p1, const float* p2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

__global__ void compute_distance_matrix_kernel(const float* points, float* distances, int n_points, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_points && j < n_points && i <= j) {
        float dist = euclidean_distance_gpu(&points[i * dim], &points[j * dim], dim);
        distances[i * n_points + j] = dist;
        distances[j * n_points + i] = dist;
    }
}

__global__ void vietoris_rips_complex_kernel(const float* distances, int* edges, float* weights, 
                                           int* edge_count, int n_points, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_points && j < n_points && i < j) {
        float dist = distances[i * n_points + j];
        if (dist <= epsilon) {
            int edge_idx = atomicAdd(edge_count, 1);
            edges[edge_idx * 2] = i;
            edges[edge_idx * 2 + 1] = j;
            weights[edge_idx] = dist;
        }
    }
}

__global__ void simplicial_complex_construction_kernel(const int* edges, const float* weights, 
                                                     int* triangles, float* triangle_weights,
                                                     int* triangle_count, int n_edges, 
                                                     const float* distances, int n_points, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_edges) {
        int v1 = edges[idx * 2];
        int v2 = edges[idx * 2 + 1];
        
        for (int i = 0; i < n_points; i++) {
            if (i != v1 && i != v2) {
                float d1 = distances[v1 * n_points + i];
                float d2 = distances[v2 * n_points + i];
                
                if (d1 <= epsilon && d2 <= epsilon) {
                    int tri_idx = atomicAdd(triangle_count, 1);
                    triangles[tri_idx * 3] = v1;
                    triangles[tri_idx * 3 + 1] = v2;
                    triangles[tri_idx * 3 + 2] = i;
                    triangle_weights[tri_idx] = fmaxf(fmaxf(weights[idx], d1), d2);
                }
            }
        }
    }
}

__global__ void boundary_matrix_kernel(const int* simplices, int* boundary_matrix, 
                                     int n_simplices, int simplex_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_simplices) {
        for (int i = 0; i <= simplex_dim; i++) {
            int sign = (i % 2 == 0) ? 1 : -1;
            int boundary_simplex[8];
            int k = 0;
            
            for (int j = 0; j <= simplex_dim; j++) {
                if (j != i) {
                    boundary_simplex[k++] = simplices[idx * (simplex_dim + 1) + j];
                }
            }
            
            for (int row = 0; row < n_simplices; row++) {
                bool match = true;
                for (int j = 0; j < simplex_dim; j++) {
                    if (simplices[row * simplex_dim + j] != boundary_simplex[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    boundary_matrix[row * n_simplices + idx] = sign;
                    break;
                }
            }
        }
    }
}

__global__ void persistent_homology_reduction_kernel(int* matrix, int* low, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols) {
        for (int row = rows - 1; row >= 0; row--) {
            if (matrix[row * cols + col] != 0) {
                low[col] = row;
                break;
            }
        }
        
        for (int other_col = 0; other_col < col; other_col++) {
            if (low[other_col] == low[col] && low[col] != -1) {
                for (int row = 0; row < rows; row++) {
                    matrix[row * cols + col] ^= matrix[row * cols + other_col];
                }
                
                low[col] = -1;
                for (int row = rows - 1; row >= 0; row--) {
                    if (matrix[row * cols + col] != 0) {
                        low[col] = row;
                        break;
                    }
                }
            }
        }
    }
}

__global__ void persistence_pairs_kernel(const int* low, const float* weights, 
                                        float* birth_death_pairs, int* pair_count, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < cols && low[col] != -1) {
        int pair_idx = atomicAdd(pair_count, 1);
        birth_death_pairs[pair_idx * 2] = weights[low[col]];
        birth_death_pairs[pair_idx * 2 + 1] = weights[col];
    }
}

__global__ void parallel_reduction_kernel(float* data, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? data[i] : 0;
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) data[blockIdx.x] = sdata[0];
}

__global__ void sparse_matrix_multiply_kernel(const int* csr_row_ptr, const int* csr_col_idx, 
                                            const float* csr_values, const float* dense_vector,
                                            float* result, int n_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n_rows) {
        float sum = 0.0f;
        int start = csr_row_ptr[row];
        int end = csr_row_ptr[row + 1];
        
        for (int j = start; j < end; j++) {
            sum += csr_values[j] * dense_vector[csr_col_idx[j]];
        }
        
        result[row] = sum;
    }
}

__global__ void memory_coalesced_transpose_kernel(const float* input, float* output, 
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

extern "C" {

void cuda_compute_persistence_homology(float* h_points, int n_points, int dim, 
                                     float epsilon, float* h_birth_death_pairs, 
                                     int* h_num_pairs) {
    
    float *d_points, *d_distances, *d_weights, *d_birth_death_pairs;
    int *d_edges, *d_triangles, *d_boundary_matrix, *d_low;
    int *d_edge_count, *d_triangle_count, *d_pair_count;
    
    size_t points_size = n_points * dim * sizeof(float);
    size_t distances_size = n_points * n_points * sizeof(float);
    size_t max_edges = n_points * (n_points - 1) / 2;
    size_t max_triangles = n_points * (n_points - 1) * (n_points - 2) / 6;
    
    cudaMalloc(&d_points, points_size);
    cudaMalloc(&d_distances, distances_size);
    cudaMalloc(&d_edges, max_edges * 2 * sizeof(int));
    cudaMalloc(&d_weights, max_edges * sizeof(float));
    cudaMalloc(&d_triangles, max_triangles * 3 * sizeof(int));
    cudaMalloc(&d_boundary_matrix, max_triangles * max_edges * sizeof(int));
    cudaMalloc(&d_low, max_edges * sizeof(int));
    cudaMalloc(&d_birth_death_pairs, max_edges * 2 * sizeof(float));
    cudaMalloc(&d_edge_count, sizeof(int));
    cudaMalloc(&d_triangle_count, sizeof(int));
    cudaMalloc(&d_pair_count, sizeof(int));
    
    cudaMemcpy(d_points, h_points, points_size, cudaMemcpyHostToDevice);
    cudaMemset(d_edge_count, 0, sizeof(int));
    cudaMemset(d_triangle_count, 0, sizeof(int));
    cudaMemset(d_pair_count, 0, sizeof(int));
    
    dim3 block_size(16, 16);
    dim3 grid_size((n_points + block_size.x - 1) / block_size.x, 
                   (n_points + block_size.y - 1) / block_size.y);
    
    compute_distance_matrix_kernel<<<grid_size, block_size>>>(d_points, d_distances, n_points, dim);
    
    vietoris_rips_complex_kernel<<<grid_size, block_size>>>(d_distances, d_edges, d_weights, 
                                                          d_edge_count, n_points, epsilon);
    
    int h_edge_count;
    cudaMemcpy(&h_edge_count, d_edge_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    dim3 edge_block_size(256);
    dim3 edge_grid_size((h_edge_count + edge_block_size.x - 1) / edge_block_size.x);
    
    simplicial_complex_construction_kernel<<<edge_grid_size, edge_block_size>>>(
        d_edges, d_weights, d_triangles, d_weights, d_triangle_count, 
        h_edge_count, d_distances, n_points, epsilon);
    
    boundary_matrix_kernel<<<edge_grid_size, edge_block_size>>>(d_edges, d_boundary_matrix, 
                                                              h_edge_count, 1);
    
    persistent_homology_reduction_kernel<<<edge_grid_size, edge_block_size>>>(
        d_boundary_matrix, d_low, h_edge_count, h_edge_count);
    
    persistence_pairs_kernel<<<edge_grid_size, edge_block_size>>>(d_low, d_weights, 
                                                                d_birth_death_pairs, 
                                                                d_pair_count, h_edge_count);
    
    cudaMemcpy(h_num_pairs, d_pair_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_birth_death_pairs, d_birth_death_pairs, 
               (*h_num_pairs) * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_points);
    cudaFree(d_distances);
    cudaFree(d_edges);
    cudaFree(d_weights);
    cudaFree(d_triangles);
    cudaFree(d_boundary_matrix);
    cudaFree(d_low);
    cudaFree(d_birth_death_pairs);
    cudaFree(d_edge_count);
    cudaFree(d_triangle_count);
    cudaFree(d_pair_count);
}

} 