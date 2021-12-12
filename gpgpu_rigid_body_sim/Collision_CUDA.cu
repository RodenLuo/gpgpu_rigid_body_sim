#include "Collision_CUDA.cuh"
#include <iostream>

__device__ glm::vec3 vload3(int i, float* array)
{
	return glm::vec3(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
}

__device__ void vstore3(glm::vec3 vec, int i, float* array)
{
	array[i * 3] = vec.x;
	array[i * 3 + 1] = vec.y;
	array[i * 3 + 2] = vec.z;
}

__global__ void collision_kernel(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, glm::vec3 gravity, int ballCollisionRun,
	glm::vec3 barrierShift, int barrierIsOn)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	glm::vec3 iVelocity = vload3(i, velocities);
	glm::vec3 iPosition = vload3(i, positions);

	iPosition += 1.0;

	vstore3(iPosition, i, positions);


}


void collision_kernel_call(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, glm::vec3 gravity, int ballCollisionRun,
	glm::vec3 barrierShift, int barrierIsOn, dim3 grid_dim, dim3 block_dim)
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