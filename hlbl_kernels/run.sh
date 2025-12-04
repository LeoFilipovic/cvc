#!/bin/bash
#SBATCH --job-name=compute_2p2_gpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --account=lp142
#SBATCH --partition=normal
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log

# 1. Enable CUDA-Aware MPI (Critical for GH200 Performance)
# This tells Cray MPI to use GPU pointers directly.
export MPICH_GPU_SUPPORT_ENABLED=1

# 2. Optimization variables for GTL (GPU Transport Layer)
export MPICH_OFI_NIC_POLICY=GPU
# Ensures MPI uses the NIC closest to the GPU
export MPICH_OFI_NIC_VERBOSE=1 

# 3. Bindings
# Bind MPI ranks to the specific GPU/Grace-CPU pair
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on hosts: $(hostname)"
echo "CUDA-Aware MPI Enabled: $MPICH_GPU_SUPPORT_ENABLED"

# 5. Run
# -u : unbuffered output
# --gpu-bind=closest : ensures the rank uses the GPU attached to its CPU socket
#srun -u --gpu-bind=closest ./kernels


# 5.1 Run with ncu
#OUTPUT_DIR="ncu_report_${SLURM_JOB_ID}"
#mkdir -p $OUTPUT_DIR
# --target-processes all profile all processes concurrently
# --force-overwrite overwrite existing report files
# --set full profile a comprehensive set of metric
# --launch-count 1 launch only once
srun -u --gpu-bind=closest \
    ncu \
    --target-processes all \
    --force-overwrite \
    --set full \
    --launch-count 1 \
    ./kernels