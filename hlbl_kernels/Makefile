CXX      := g++
NVCC     := nvcc

# Architecture for Grace Hopper (GH200) is sm_90
ARCH     := -arch=sm_90

# Standard flags
CXXFLAGS := -O2 -std=c++17 -fopenmp
NVFLAGS  := -O2 -ccbin=g++ $(ARCH) --relocatable-device-code=true -Xcompiler -fopenmp --ptxas-options=-v

# --- PATHS ---
# CSCS typically sets CRAY_MPICH_DIR or MPICH_DIR when the module is loaded.
# We use that to find MPI headers if nvcc doesn't find them automatically.
MPI_INC_DIR := $(CUDA_HOME)/include
MPI_LIB_DIR := $(CUDA_HOME)/lib

# Your custom libraries
CPU_INC  := ../../build_kqed_cpu/include
CPU_LIB  := ../../build_kqed_cpu/lib
GPU_INC  := ../../build_kqed_gpu/install_dir/include
GPU_LIB  := ../../build_kqed_gpu/install_dir/lib

# --- LIBRARIES ---
# Link against MPI and your custom libraries
CPU_LD_FLAGS := -L$(MPI_LIB_DIR) -lmpi -L$(CPU_LIB) -lKQED
GPU_LD_FLAGS := -L$(MPI_LIB_DIR) -lmpi -L$(GPU_LIB) -lKQED_cuda

# Targets
all: test_gpu test

test_gpu: test_gpu.cu
	$(NVCC) $(NVFLAGS) test_gpu.cu kernels.cu global.cpp -I$(MPI_INC_DIR) -I$(GPU_INC) $(GPU_LD_FLAGS) -o test_gpu

test: test.cpp
	$(CXX) $(CXXFLAGS) test.cpp kernels.cpp global.cpp -I$(MPI_INC_DIR) -I$(CPU_INC) $(CPU_LD_FLAGS) -o test


clean:
	rm -f test_gpu test

.PHONY: all clean
