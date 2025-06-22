#include <iostream>
#include <iomanip>

extern "C" {
    void run_comprehensive_benchmark();
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "KAN-TDA GPU Performance Benchmark Suite\n";
    std::cout << "=======================================\n\n";
    
    run_comprehensive_benchmark();
    
    return 0;
} 