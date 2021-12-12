#include "vec_add_test.cuh"
#include <iostream>


__global__ void vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

void vecAdd_call(int* A, int* B, int* C, int N, dim3 block_no, dim3 block_size)
{
	vecAdd <<<block_no, block_size >>> (A, B, C, N);
	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}