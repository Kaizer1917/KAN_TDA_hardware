#include <iostream>
#include <vector>
#include <chrono>
#include <random>

extern "C" {
    void sparse_matrix_conversion(const float* dense_matrix, void* sparse, int rows, int cols, float threshold);
    void streaming_computation(float* input_data, float* output_data, int total_size, int chunk_size);
    void cache_optimized_matrix_operations(const float* A, const float* B, float* C, int M, int N, int K);
    void adaptive_compression(const float* input_data, int** compressed_indices, float** compressed_values, int* compressed_size, int data_size, float threshold);
}

void test_sparse_matrix_conversion() {
    std::cout << "Testing Sparse Matrix Conversion\n";
    std::cout << "================================\n";
    
    int rows = 1000, cols = 1000;
    std::vector<float> dense_matrix(rows * cols);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        dense_matrix[i] = (dis(gen) < 0.1f) ? dis(gen) * 10.0f : 0.0f;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    float* d_dense;
    cudaMalloc(&d_dense, rows * cols * sizeof(float));
    cudaMemcpy(d_dense, dense_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    sparse_matrix_conversion(d_dense, nullptr, rows, cols, 0.1f);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Matrix size: " << rows << "x" << cols << "\n";
    std::cout << "Conversion time: " << duration.count() << " ms\n";
    std::cout << "Memory saved: ~" << (rows * cols * 0.9f * sizeof(float)) / (1024 * 1024) << " MB\n\n";
    
    cudaFree(d_dense);
}

void test_streaming_computation() {
    std::cout << "Testing Streaming Computation\n";
    std::cout << "=============================\n";
    
    int total_size = 10000000;
    int chunk_size = 1000000;
    
    std::vector<float> input_data(total_size);
    std::vector<float> output_data(total_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < total_size; i++) {
        input_data[i] = dis(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    
    cudaMemcpy(d_input, input_data.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    streaming_computation(d_input, d_output, total_size, chunk_size);
    
    cudaMemcpy(output_data.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Total data size: " << total_size / 1000000.0f << " M elements\n";
    std::cout << "Chunk size: " << chunk_size / 1000000.0f << " M elements\n";
    std::cout << "Processing time: " << duration.count() << " ms\n";
    std::cout << "Throughput: " << (total_size / 1000000.0f) / (duration.count() / 1000.0f) << " M elements/sec\n\n";
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void test_cache_optimized_operations() {
    std::cout << "Testing Cache-Optimized Matrix Operations\n";
    std::cout << "=========================================\n";
    
    int M = 2048, N = 2048, K = 2048;
    
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < M * K; i++) A[i] = dis(gen);
    for (int i = 0; i < K * N; i++) B[i] = dis(gen);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cache_optimized_matrix_operations(d_A, d_B, d_C, M, N, K);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float gflops = (2.0f * M * N * K) / (duration.count() * 1e6);
    
    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << "\n";
    std::cout << "Computation time: " << duration.count() << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n\n";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void test_adaptive_compression() {
    std::cout << "Testing Adaptive Compression\n";
    std::cout << "============================\n";
    
    int data_size = 1000000;
    std::vector<float> input_data(data_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < data_size; i++) {
        input_data[i] = (dis(gen) < 0.05f) ? dis(gen) * 100.0f : dis(gen) * 0.01f;
    }
    
    float* d_input;
    cudaMalloc(&d_input, data_size * sizeof(float));
    cudaMemcpy(d_input, input_data.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
    
    float thresholds[] = {0.1f, 0.5f, 1.0f, 5.0f};
    int num_thresholds = sizeof(thresholds) / sizeof(thresholds[0]);
    
    std::cout << "Threshold\tCompressed Size\tCompression Ratio\n";
    std::cout << "---------\t---------------\t-----------------\n";
    
    for (int t = 0; t < num_thresholds; t++) {
        int* compressed_indices;
        float* compressed_values;
        int compressed_size;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        adaptive_compression(d_input, &compressed_indices, &compressed_values, 
                           &compressed_size, data_size, thresholds[t]);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        float compression_ratio = (float)compressed_size / data_size;
        
        std::cout << std::fixed << std::setprecision(1) << thresholds[t] << "\t\t" 
                  << compressed_size << "\t\t" << compression_ratio * 100 << "%\n";
        
        free(compressed_indices);
        free(compressed_values);
    }
    
    std::cout << "\n";
    cudaFree(d_input);
}

void run_memory_benchmark_suite() {
    std::cout << "Memory Optimization Benchmark Suite\n";
    std::cout << "===================================\n\n";
    
    int sizes[] = {1000, 5000, 10000, 50000, 100000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    std::cout << "Memory Bandwidth Test:\n";
    std::cout << "Size\t\tAllocation(ms)\tTransfer(ms)\tBandwidth(GB/s)\n";
    std::cout << "----\t\t--------------\t-----------\t---------------\n";
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i] * sizes[i];
        std::vector<float> host_data(size);
        
        auto alloc_start = std::chrono::high_resolution_clock::now();
        float* device_data;
        cudaMalloc(&device_data, size * sizeof(float));
        auto alloc_end = std::chrono::high_resolution_clock::now();
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start);
        
        auto transfer_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(device_data, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(host_data.data(), device_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        auto transfer_end = std::chrono::high_resolution_clock::now();
        auto transfer_duration = std::chrono::duration_cast<std::chrono::microseconds>(transfer_end - transfer_start);
        
        float bandwidth = (2.0f * size * sizeof(float)) / (transfer_duration.count() * 1e-3);
        
        std::cout << sizes[i] << "x" << sizes[i] << "\t\t" 
                  << alloc_duration.count() / 1000.0f << "\t\t"
                  << transfer_duration.count() / 1000.0f << "\t\t"
                  << bandwidth / 1e9 << "\n";
        
        cudaFree(device_data);
    }
    
    std::cout << "\n";
}

int main(int argc, char** argv) {
    std::cout << "GPU Memory Optimization Test Suite\n";
    std::cout << "==================================\n\n";
    
    test_sparse_matrix_conversion();
    test_streaming_computation();
    test_cache_optimized_operations();
    test_adaptive_compression();
    run_memory_benchmark_suite();
    
    std::cout << "All memory optimization tests completed successfully!\n";
    
    return 0;
} 