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

	if (i > numberOfBalls)
		return;

	glm::vec3 iVelocity = vload3(i, velocities);
	glm::vec3 iPosition = vload3(i, positions);

	bool tmp_test = false;
	if (tmp_test)
	{
		iPosition += 0.1f;
		vstore3(iPosition, i, positions);
		return;
	}

	float dt = 1.0;

	//Apply the velocities, the gravity and the resistance
	iVelocity = iVelocity - gravity;
	iVelocity = iVelocity * resistance;
	iPosition = iPosition + iVelocity * dt;

	//Handle the ball to ball collision and reflection
	if (ballCollisionRun == 1) {
		for (size_t j = 0; j < numberOfBalls; j++) {
			if (i != j) {
				glm::vec3 jVelocity = vload3(j, velocities);
				glm::vec3 jPosition = vload3(j, positions);
				float dist = glm::distance(iPosition, jPosition);
				if (dist < 2.0f) {
					//Becouse zero division.
					if (dist <= 0.1f) {
						iPosition = iPosition + iVelocity + 2.0f;
						jPosition = jPosition + jVelocity + 2.0f;
					}
					float iLength = glm::length(iVelocity);
					glm::vec3 p = iPosition - jPosition;
					glm::vec3 v = iVelocity - jVelocity;
					iVelocity = glm::normalize(iVelocity - (glm::dot(v, p)) / (sqrt(v.x * v.x + v.y * v.y + v.z * v.z)) * p) * (iLength / 2.0f);
					float jLength = glm::length(jVelocity);
					p = jPosition - iPosition;
					v = jVelocity - iVelocity;
					jVelocity = glm::normalize(jVelocity - (glm::dot(v, p)) / (sqrt(v.x * v.x + v.y * v.y + v.z * v.z)) * p) * (jLength / 2.0f);

					iVelocity = iVelocity * (resistance * resistance); //Plus resistance
					jVelocity = jVelocity * (resistance * resistance); //Plus resistance
					iPosition = iPosition + (((2.2f - dist) / glm::length(iVelocity)) * iVelocity);
					jPosition = jPosition - (((2.2f - dist) / glm::length(jVelocity)) * jVelocity);

					vstore3(jVelocity, j, velocities);
					vstore3(jPosition, j, positions);
				}
			}
		}
	}

	//Handle the ball to barrier collision and refraction
	if (barrierIsOn == 1) {
		glm::vec3 normal2 = glm::vec3(0.0f, 0.0f, 0.0f);
		float barrierSize = boxSize / 6;

		if (iPosition.y <= (-boxSize + barrierSize * 2.0f) + 1.0f &&
			(iPosition.x >= -barrierSize + barrierShift.x) && (iPosition.x <= barrierSize + barrierShift.x) &&
			(iPosition.z >= -barrierSize + barrierShift.z) && (iPosition.z <= barrierSize + +barrierShift.z)) {
			normal2 = glm::vec3(0.0f, 1.0f, 0.0f);
			iPosition.y = (-boxSize + barrierSize * 2.0f) + 1.01f;
		}
		if (iPosition.y < (-boxSize + barrierSize * 2.0f) &&
			(iPosition.x > -barrierSize + barrierShift.x - 1.0f) && (iPosition.x < -barrierSize + barrierShift.x) &&
			(iPosition.z > -barrierSize + barrierShift.z) && (iPosition.z < barrierSize + +barrierShift.z)) {
			normal2 = glm::vec3(1.0f, 0.0f, 0.0f);
			iPosition.x = -barrierSize + barrierShift.x - 1.01f;
		}
		if (iPosition.y < (-boxSize + barrierSize * 2.0f) &&
			(iPosition.x < barrierSize + barrierShift.x + 1.0f) && (iPosition.x > barrierSize + barrierShift.x) &&
			(iPosition.z > -barrierSize + barrierShift.z) && (iPosition.z < barrierSize + +barrierShift.z)) {
			normal2 = glm::vec3(-1.0f, 0.0f, 0.0f);
			iPosition.x = barrierSize + barrierShift.x + 1.02f;
		}

		if (iPosition.y < (-boxSize + barrierSize * 2.0f) &&
			(iPosition.z > -barrierSize + barrierShift.z - 1.0f) && (iPosition.z < -barrierSize + barrierShift.z) &&
			(iPosition.x > -barrierSize + barrierShift.x) && (iPosition.x < barrierSize + +barrierShift.x)) {
			normal2 = glm::vec3(0.0f, 0.0f, 1.0f);
			iPosition.z = -barrierSize + barrierShift.z - 1.01f;
		}
		if (iPosition.y < (-boxSize + barrierSize * 2.0f) &&
			(iPosition.z < barrierSize + barrierShift.z + 1.0f) && (iPosition.z > barrierSize + barrierShift.z) &&
			(iPosition.x > -barrierSize + barrierShift.x) && (iPosition.x < barrierSize + +barrierShift.x)) {
			normal2 = glm::vec3(0.0f, 0.0f, -1.0f);
			iPosition.z = barrierSize + barrierShift.z + 1.02f;
		}

		if (normal2.x != 0.0f || normal2.y != 0.0f || normal2.z != 0.0f) {
			iVelocity = iVelocity - 2.0f * glm::dot(normal2, iVelocity) * normal2;
			iVelocity = ((iVelocity * resistance) * resistance) * resistance; //Plus resistance
		}
	}

	//Handle the ball to wall collision and reflection
	glm::vec3 normal = glm::vec3(0.0f, 0.0f, 0.0f);

	if (iPosition.y <= -boxSize + 1.0f) {
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
		iPosition.y = -boxSize + 1.01f;
	}
	if (iPosition.y >= boxSize - 1.0f) {
		normal = glm::vec3(0.0f, -1.0f, 0.0f);
		iPosition.y = boxSize - 1.01f;
	}
	if (iPosition.x <= -boxSize + 1.0f) {
		normal = glm::vec3(1.0f, 0.0f, 0.0f);
		iPosition.x = -boxSize + 1.01f;
	}
	if (iPosition.x >= boxSize - 1.0f) {
		normal = glm::vec3(-1.0f, 0.0f, 0.0f);
		iPosition.x = boxSize - 1.01f;
	}
	if (iPosition.z <= -boxSize + 1.0f) {
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
		iPosition.z = -boxSize + 1.01f;
	}
	if (iPosition.z >= boxSize - 1.0f) {
		normal = glm::vec3(0.0f, 0.0f, -1.0f);
		iPosition.z = boxSize - 1.01f;
	}

	if (normal.x != 0.0f || normal.y != 0.0f || normal.z != 0.0f) {
		iVelocity = iVelocity - 2.0f * glm::dot(normal, iVelocity) * normal;
		iVelocity = ((iVelocity * resistance) * resistance) * resistance; //Plus resistance
	}

	vstore3(iVelocity, i, velocities);
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