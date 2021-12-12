#include "Collision_CUDA.cuh"
#include <iostream>


__global__ void collision_kernel(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, float* gravity, int ballCollisionRun,
	float* barrierShift, int barrierIsOn)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	positions[3 * i] = 0.0f;
}


void collision_kernel_call(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, float* gravity, int ballCollisionRun,
	float* barrierShift, int barrierIsOn, dim3 grid_dim, dim3 block_dim)
{
	collision_kernel <<<grid_dim, block_dim>>> (positions, velocities, numberOfBalls,
		boxSize, resistance, gravity, ballCollisionRun,
		barrierShift, barrierIsOn);
	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}