#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>

__global__ void collision_kernel(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, glm::vec3 gravity, int ballCollisionRun,
	glm::vec3 barrierShift, int barrierIsOn);

void collision_kernel_call(float* positions, float* velocities, int numberOfBalls,
	float boxSize, float resistance, glm::vec3 gravity, int ballCollisionRun,
	glm::vec3 barrierShift, int barrierIsOn, dim3 grid_dim, dim3 block_dim);

