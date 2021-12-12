#pragma once

#include <cuda_runtime.h>

__global__ void collision_kernel(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, float* gravity, int ballCollisionRun,
	float* barrierShift, int barrierIsOn);

void collision_kernel_call(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, float* gravity, int ballCollisionRun,
	float* barrierShift, int barrierIsOn, dim3 grid_dim, dim3 block_dim);

