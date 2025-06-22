NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_70 -Xcompiler -fopenmp -Xcompiler -march=native
MPI_CFLAGS = $(shell mpi-config --cflags)
MPI_LIBS = $(shell mpi-config --libs)

CUDA_LIBS = -lcuda -lcudart -lcublas -lcusparse -lcurand -lnccl
SYSTEM_LIBS = -lm -lpthread

INCLUDES = -I/usr/local/cuda/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/include

SOURCES = cuda_persistent_homology.cu multi_gpu_architecture.cu memory_optimization.cu gpu_benchmark.cu
OBJECTS = $(SOURCES:.cu=.o)
TARGETS = kan_tda_gpu kan_tda_multi_gpu kan_tda_benchmark kan_tda_memory_test

all: $(TARGETS)

kan_tda_gpu: cuda_persistent_homology.o
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUDA_LIBS) $(SYSTEM_LIBS)

kan_tda_multi_gpu: multi_gpu_architecture.o cuda_persistent_homology.o
	mpicxx $(MPI_CFLAGS) -o $@ $^ $(CUDA_LIBS) $(MPI_LIBS) $(SYSTEM_LIBS) -lcuda

kan_tda_benchmark: gpu_benchmark.o
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUDA_LIBS) $(SYSTEM_LIBS)

kan_tda_memory_test: memory_optimization.o
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(CUDA_LIBS) $(SYSTEM_LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

test_single_gpu: kan_tda_gpu
	./kan_tda_gpu 10000 3 0.5

test_multi_gpu: kan_tda_multi_gpu
	mpirun -np 4 ./kan_tda_multi_gpu 50000 3 0.5

test_benchmark: kan_tda_benchmark
	./kan_tda_benchmark

test_memory: kan_tda_memory_test
	./kan_tda_memory_test

test_all: test_single_gpu test_multi_gpu test_benchmark test_memory

clean:
	rm -f $(OBJECTS) $(TARGETS)

install: $(TARGETS)
	mkdir -p ../../../bin
	cp $(TARGETS) ../../../bin/

benchmark_suite: kan_tda_benchmark
	./kan_tda_benchmark > benchmark_results.txt
	echo "Benchmark results saved to benchmark_results.txt"

profile_gpu: kan_tda_gpu
	nvprof --print-gpu-trace ./kan_tda_gpu 10000 3 0.5

profile_memory: kan_tda_memory_test
	nvprof --unified-memory-profiling off ./kan_tda_memory_test

performance_analysis: kan_tda_benchmark
	ncu --set full ./kan_tda_benchmark

distributed_test: kan_tda_multi_gpu
	mpirun -np 2 --hostfile hostfile ./kan_tda_multi_gpu 100000 3 0.5

stress_test: kan_tda_gpu kan_tda_benchmark
	for size in 1000 10000 100000 1000000; do \
		echo "Testing size: $$size"; \
		./kan_tda_gpu $$size 3 0.5; \
	done

memory_stress_test: kan_tda_memory_test
	for thresh in 0.1 0.01 0.001; do \
		echo "Testing compression threshold: $$thresh"; \
		./kan_tda_memory_test $$thresh; \
	done

gpu_cluster_test: kan_tda_multi_gpu
	mpirun -np 8 --map-by ppr:2:node ./kan_tda_multi_gpu 500000 3 0.5

.PHONY: all clean install test_single_gpu test_multi_gpu test_benchmark test_memory test_all benchmark_suite profile_gpu profile_memory performance_analysis distributed_test stress_test memory_stress_test gpu_cluster_test 