#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

extern "C" {
    void cuda_compute_persistence_homology(float* h_points, int n_points, int dim, 
                                         float epsilon, float* h_birth_death_pairs, 
                                         int* h_num_pairs);
}

void generate_random_points(std::vector<float>& points, int n_points, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    
    points.resize(n_points * dim);
    for (int i = 0; i < n_points * dim; i++) {
        points[i] = dis(gen);
    }
}

void generate_torus_points(std::vector<float>& points, int n_points) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dis(0.0f, 2.0f * M_PI);
    
    points.resize(n_points * 3);
    float R = 3.0f;
    float r = 1.0f;
    
    for (int i = 0; i < n_points; i++) {
        float u = angle_dis(gen);
        float v = angle_dis(gen);
        
        points[i * 3 + 0] = (R + r * cosf(v)) * cosf(u);
        points[i * 3 + 1] = (R + r * cosf(v)) * sinf(u);
        points[i * 3 + 2] = r * sinf(v);
    }
}

void generate_sphere_points(std::vector<float>& points, int n_points) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dis(0.0f, 1.0f);
    
    points.resize(n_points * 3);
    
    for (int i = 0; i < n_points; i++) {
        float x = normal_dis(gen);
        float y = normal_dis(gen);
        float z = normal_dis(gen);
        
        float norm = sqrtf(x*x + y*y + z*z);
        
        points[i * 3 + 0] = x / norm;
        points[i * 3 + 1] = y / norm;
        points[i * 3 + 2] = z / norm;
    }
}

void print_persistence_diagram(const std::vector<float>& birth_death_pairs, int num_pairs) {
    std::cout << "Persistence Diagram (" << num_pairs << " pairs):\n";
    std::cout << "Birth\t\tDeath\t\tLifetime\n";
    std::cout << "-----\t\t-----\t\t--------\n";
    
    for (int i = 0; i < num_pairs; i++) {
        float birth = birth_death_pairs[i * 2];
        float death = birth_death_pairs[i * 2 + 1];
        float lifetime = death - birth;
        
        std::cout << birth << "\t\t" << death << "\t\t" << lifetime << "\n";
    }
}

void benchmark_performance(int n_points, int dim, float epsilon, int num_runs) {
    std::vector<float> points;
    generate_random_points(points, n_points, dim);
    
    std::vector<float> birth_death_pairs(n_points * 2);
    int num_pairs;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int run = 0; run < num_runs; run++) {
        cuda_compute_persistence_homology(points.data(), n_points, dim, epsilon, 
                                        birth_death_pairs.data(), &num_pairs);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Performance Benchmark Results:\n";
    std::cout << "Points: " << n_points << ", Dimension: " << dim << ", Epsilon: " << epsilon << "\n";
    std::cout << "Runs: " << num_runs << ", Total Time: " << duration.count() << " ms\n";
    std::cout << "Average Time per Run: " << (float)duration.count() / num_runs << " ms\n";
    std::cout << "Throughput: " << (float)(n_points * num_runs) / (duration.count() / 1000.0f) << " points/sec\n";
}

int main(int argc, char** argv) {
    int n_points = 1000;
    int dim = 3;
    float epsilon = 1.0f;
    
    if (argc >= 2) n_points = std::atoi(argv[1]);
    if (argc >= 3) dim = std::atoi(argv[2]);
    if (argc >= 4) epsilon = std::atof(argv[3]);
    
    std::cout << "CUDA Persistent Homology Computation\n";
    std::cout << "====================================\n\n";
    
    std::cout << "Test 1: Random Point Cloud\n";
    std::vector<float> random_points;
    generate_random_points(random_points, n_points, dim);
    
    std::vector<float> birth_death_pairs(n_points * 2);
    int num_pairs;
    
    auto start = std::chrono::high_resolution_clock::now();
    cuda_compute_persistence_homology(random_points.data(), n_points, dim, epsilon, 
                                    birth_death_pairs.data(), &num_pairs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Random points - Time: " << duration.count() << " ms\n";
    print_persistence_diagram(birth_death_pairs, std::min(num_pairs, 10));
    std::cout << "\n";
    
    if (dim == 3) {
        std::cout << "Test 2: Torus Point Cloud\n";
        std::vector<float> torus_points;
        generate_torus_points(torus_points, n_points);
        
        start = std::chrono::high_resolution_clock::now();
        cuda_compute_persistence_homology(torus_points.data(), n_points, 3, epsilon, 
                                        birth_death_pairs.data(), &num_pairs);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Torus points - Time: " << duration.count() << " ms\n";
        print_persistence_diagram(birth_death_pairs, std::min(num_pairs, 10));
        std::cout << "\n";
        
        std::cout << "Test 3: Sphere Point Cloud\n";
        std::vector<float> sphere_points;
        generate_sphere_points(sphere_points, n_points);
        
        start = std::chrono::high_resolution_clock::now();
        cuda_compute_persistence_homology(sphere_points.data(), n_points, 3, epsilon, 
                                        birth_death_pairs.data(), &num_pairs);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Sphere points - Time: " << duration.count() << " ms\n";
        print_persistence_diagram(birth_death_pairs, std::min(num_pairs, 10));
        std::cout << "\n";
    }
    
    std::cout << "Performance Benchmark:\n";
    benchmark_performance(n_points, dim, epsilon, 10);
    
    std::cout << "\nScalability Test:\n";
    int test_sizes[] = {100, 500, 1000, 2000, 5000};
    int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    std::cout << "Size\t\tTime(ms)\tPoints/sec\n";
    std::cout << "----\t\t--------\t----------\n";
    
    for (int i = 0; i < num_test_sizes; i++) {
        std::vector<float> test_points;
        generate_random_points(test_points, test_sizes[i], dim);
        
        start = std::chrono::high_resolution_clock::now();
        cuda_compute_persistence_homology(test_points.data(), test_sizes[i], dim, epsilon, 
                                        birth_death_pairs.data(), &num_pairs);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float throughput = (float)test_sizes[i] / (duration.count() / 1000.0f);
        std::cout << test_sizes[i] << "\t\t" << duration.count() << "\t\t" << throughput << "\n";
    }
    
    return 0;
} 