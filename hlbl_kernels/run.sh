srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=closest ./test_gpu
