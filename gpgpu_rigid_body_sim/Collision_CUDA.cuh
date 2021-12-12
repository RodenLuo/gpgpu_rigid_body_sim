#pragma once

#include <cuda_runtime.h>

__global__ void collision_kernel(int* A, int*B, int*C, int N);

void collision_kernel_call(int * A, int*B, int* C, int N, dim3 grid_dim, dim3 block_dim);

