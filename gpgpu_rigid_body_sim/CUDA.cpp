#include "Simulation.h"

#include "vec_add_test.cuh"

#include "Collision_CUDA.cuh"

void Simulation::VecAddTest_CUDA()
{
	int* a, * b;  // host data
	int* c, * c2;  // results

	printf("Begin \n");
	int n = 10000;
	int nBytes = n * sizeof(int);
	int block_size, block_no;
	a = (int*)malloc(nBytes);
	b = (int*)malloc(nBytes);
	c = (int*)malloc(nBytes);
	c2 = (int*)malloc(nBytes);
	int* a_d, * b_d, * c_d;
	block_size = 1000;
	block_no = n / block_size;
	dim3 dimBlock(block_size, 1, 1);
	dim3 dimGrid(block_no, 1, 1);
	for (int i = 0; i < n; i++)
		a[i] = i, b[i] = i;
	printf("Allocating device memory on host..\n");
	cudaMalloc((void**)&a_d, n * sizeof(int));
	cudaMalloc((void**)&b_d, n * sizeof(int));
	cudaMalloc((void**)&c_d, n * sizeof(int));
	printf("Copying to device..\n");

	cudaMemcpy(a_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, n * sizeof(int), cudaMemcpyHostToDevice);
	clock_t start_d = clock();
	printf("Doing GPU Vector add\n");

	vecAdd_call(a_d, b_d, c_d, n, dimGrid, dimBlock);

	cudaMemcpy(c, c_d, n * sizeof(int), cudaMemcpyDeviceToHost);

	printf("a: %d, %d \n", a[0], a[1]);
	printf("b: %d, %d \n", b[0], b[1]);
	printf("c: %d, %d \n", c[0], c[1]);

	clock_t end_d = clock();

	double time_d = (double)(end_d - start_d) / CLOCKS_PER_SEC;

	printf("%d %f\n", n, time_d);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	return;
}

void Simulation::Collision_CUDA()
{
	int ballCollisionRunGPU = 0;
	if (ballCollisionRun) ballCollisionRunGPU = 1;

	int barrierIsOnGPU = 0;
	if (barrierIsOn) barrierIsOnGPU = 1;

	int thread_per_block = 100;
	int block_per_grid = numberOfBallsArray / thread_per_block;
	dim3 block_dim(thread_per_block, 1, 1);
	dim3 grid_dim(block_per_grid, 1, 1);

	collision_kernel_call(positions_cuda, velocities_cuda, numberOfBalls,
		boxSize, resistance, gravity_cuda, ballCollisionRun,
		barrierShift_cuda, barrierIsOn, grid_dim, block_dim);

	cudaMemcpy(positions, positions_cuda, 3 * numberOfBallsArray * sizeof(float), cudaMemcpyDeviceToHost);

	return;
}

void Simulation::Init_CUDA()
{
	cudaMalloc((void**)&positions_cuda, 3 * numberOfBallsArray * sizeof(float));
	cudaMalloc((void**)&velocities_cuda, 3 * numberOfBallsArray * sizeof(float));

	cudaMalloc((void**)&gravity_cuda, 3 * sizeof(float));
	cudaMalloc((void**)&barrierShift_cuda, 3 * sizeof(float));

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA Init error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

void Simulation::Free_CUDA()
{
	cudaFree(positions_cuda);
	cudaFree(velocities_cuda);

	cudaFree(gravity_cuda);
	cudaFree(barrierShift_cuda);
}

void Simulation::Update_CUDA(bool updateAll) {
	//Write the current vectors to CUDA

	if (updateAll) {
		cudaMemcpy(positions_cuda, positions, 3 * numberOfBallsArray * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(velocities_cuda, velocities, 3 * numberOfBallsArray * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	//Set the current variables
	cudaMemcpy(gravity_cuda, &gravity, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(barrierShift_cuda, &barrierShift, 3 * sizeof(float), cudaMemcpyHostToDevice);
}