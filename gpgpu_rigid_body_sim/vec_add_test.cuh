#pragma once

#include <cuda_runtime.h>


__global__ void vecAdd(int* A, int* B, int* C, int N);

void vecAdd_call(int* A, int* B, int* C, int N, dim3 block_no, dim3 block_size);