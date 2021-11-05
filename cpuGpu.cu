
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

// kernels transpose/copy a tile of TILE_DIM x TILE_DIM elements
// using a TILE_DIM x BLOCK_ROWS thread block, so that each thread
// transposes TILE_DIM/BLOCK_ROWS elements. TILE_DIM must be an 
// integral multiple of BLOCK_ROWS
#define TILE_DIM 32
#define BLOCK_ROWS 8
// Number of repetitions used for timing. 
#define NUM_REPS 1

__global__ void copy(float* odata, float* idata, int width, int height, int nreps);
__global__ void transposeNaive(float* odata, float* idata, int width, int height, int nreps);
__global__ void transposeCoalesced(float* odata, float* idata, int width, int height, int nreps);
__global__ void transposeNoBankConflicts(float* odata, float* idata, int width, int height, int nreps);
__global__ void copySharedMem(float* odata, float* idata, int width, int height, int nreps);
__global__ void transposeFineGrained(float* odata, float* idata, int width, int height, int nreps);
__global__ void transposeCoarseGrained(float* odata, float* idata, int width, int height, int nreps);
__global__ void transposeDiagonal(float* odata, float* idata, int width, int height, int nreps);

void computeTransposeGold(float* gold, float* idata, const  int size_x, const  int size_y);
int comparef(float* gold, float* idata, const  int size_x, const  int size_y);

int main(int argc, char** argv)
{
	// set matrix size
	const int size_x = 256, size_y = 256;
	// kernel pointer and descriptor
	void (*kernel)(float*, float*, int, int, int);
	char* kernelName;
	// execution configuration parameters
	dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM), threads(TILE_DIM, BLOCK_ROWS);
	// CUDA events
	cudaEvent_t start, stop;
	// size of memory required to store the matrix
	const int mem_size = sizeof(float) * size_x * size_y;
	// allocate host memory
	float* h_idata = (float*)malloc(mem_size);
	float* h_odata = (float*)malloc(mem_size);
	float* transposeGold = (float*)malloc(mem_size);
	float* gold;
	// allocate device memory
	float* d_idata, * d_odata;
	cudaMalloc((void**)&d_idata, mem_size);
	cudaMalloc((void**)&d_odata, mem_size);
	// initalize host data
	for (int i = 0; i < (size_x * size_y); ++i)
		h_idata[i] = (float)i;

	// copy host data to device
	cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
	// Compute reference transpose solution
	auto start_time = std::chrono::high_resolution_clock::now();
	computeTransposeGold(transposeGold, h_idata, size_x, size_y);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto time = end_time - start_time;
	// print out common data for all kernels
	printf("\nMatrix size: %dx%d, tile: %dx%d, block: %dx%d\n\n",
		size_x, size_y, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

	printf("Kernel\t\t\tLoop over kernel\tLoop within kernel\n");
	printf("------\t\t\t----------------\t------------------\n");
	//
	// loop over different kernels
	//
	for (int k = 0; k < 8; k++) {
		// set kernel pointer
		switch (k) {
		case 0:
			kernel = &copy;
			kernelName = "simple copy "; break;
		case 1:
			kernel = &copySharedMem;
			kernelName = "shared memory copy "; break;
		case 2:
			kernel = &transposeNaive;
			kernelName = "naive transpose "; break;
		case 3:
			kernel = &transposeCoalesced;
			kernelName = "coalesced transpose "; break;
		case 4:
			kernel = &transposeNoBankConflicts;
			kernelName = "no bank conflict trans"; break;
		case 5:
			kernel = &transposeCoarseGrained;
			kernelName = "coarse-grained "; break;
		case 6:
			kernel = &transposeFineGrained;
			kernelName = "fine-grained "; break;
		case 7:
			kernel = &transposeDiagonal;
			kernelName = "diagonal transpose "; break;
		}
		// set reference solution
		// NB: fine- and coarse-grained kernels are not full
		// transposes, so bypass check
		if (kernel == &copy || kernel == &copySharedMem) {
			gold = h_idata;
		}
		else if (kernel == &transposeCoarseGrained ||
			kernel == &transposeFineGrained) {
			gold = h_odata;
		}
		else {
			gold = transposeGold;
		}

		// initialize events, EC parameters
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// warmup to avoid timing startup
		kernel << <grid, threads >> > (d_odata, d_idata, size_x, size_y, 1);
		// take measurements for loop over kernel launches
		cudaEventRecord(start, 0);
		for (int i = 0; i < NUM_REPS; i++) {
			kernel << <grid, threads >> > (d_odata, d_idata, size_x, size_y, 1);
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float outerTime;
		cudaEventElapsedTime(&outerTime, start, stop);
		cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
		int res = comparef(gold, h_odata, size_x, size_y);
		// int res = 1;
		if (res != 1)
			printf("*** %s kernel FAILED ***\n", kernelName);
		// take measurements for loop inside kernel
		cudaEventRecord(start, 0);
		kernel << <grid, threads >> >
			(d_odata, d_idata, size_x, size_y, NUM_REPS);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float innerTime;
		cudaEventElapsedTime(&innerTime, start, stop);
		cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
		res = comparef(gold, h_odata, size_x, size_y);
		if (res != 1)
			printf("*** %s kernel FAILED ***\n", kernelName);

		// report time in us
		printf("%s\t%f us\t\t%f us\n",
			kernelName, outerTime * 1000, innerTime * 1000);
	}

	printf("\n CPU Transposed Matrix: \n");
	std::cout << "\nTime taken by CPU using C++ chronous lib is  " << time / std::chrono::microseconds(1) << " us" << std::endl;

	// cleanup
	free(h_idata); free(h_odata); free(transposeGold);
	cudaFree(d_idata); cudaFree(d_odata);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	return 0;
}

void computeTransposeGold(float* gold, float* idata, const  int size_x, const  int size_y)
{
	for (int y = 0; y < size_y; ++y) {
		for (int x = 0; x < size_x; ++x) {
			gold[(x * size_y) + y] = idata[(y * size_x) + x];
		}
	}
}

int comparef(float* gold, float* idata, const  int size_x, const  int size_y)
{
	for (int y = 0; y < size_y; ++y) {
		for (int x = 0; x < size_x; ++x) {
			if (gold[(x * size_y) + y] != idata[(x * size_y) + y]) return 0;
		}
	}
	return 1;
}


__global__ void copy(float* odata, float* idata, int width,
	int height, int nreps)
{
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index = xIndex + width * yIndex;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index + i * width] = idata[index + i * width];
		}
	}
}

__global__ void transposeNaive(float* odata, float* idata,
	int width, int height, int nreps)
{
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + width * yIndex;
	int index_out = yIndex + height * xIndex;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index_out + i] = idata[index_in + i * width];
		}
	}
}

__global__ void transposeCoalesced(float* odata,
	float* idata, int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			tile[threadIdx.y + i][threadIdx.x] =
				idata[index_in + i * width];
		}

		__syncthreads();

		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index_out + i * height] =
				tile[threadIdx.x][threadIdx.y + i];
		}
	}
}

__global__ void transposeNoBankConflicts(float* odata,
	float* idata, int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			tile[threadIdx.y + i][threadIdx.x] =
				idata[index_in + i * width];
		}

		__syncthreads();

		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index_out + i * height] =
				tile[threadIdx.x][threadIdx.y + i];
		}
	}
}

__global__ void copySharedMem(float* odata, float* idata,
	int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int index = xIndex + width * yIndex;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			tile[threadIdx.y + i][threadIdx.x] =
				idata[index + i * width];
		}

		__syncthreads();

		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index + i * width] =
				tile[threadIdx.y + i][threadIdx.x];
		}
	}
}

__global__ void transposeFineGrained(float* odata,
	float* idata, int width, int height, int nreps)
{
	__shared__ float block[TILE_DIM][TILE_DIM + 1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index = xIndex + (yIndex)*width;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			block[threadIdx.y + i][threadIdx.x] =
				idata[index + i * width];
		}

		__syncthreads();
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index + i * height] =
				block[threadIdx.x][threadIdx.y + i];
		}
	}
}

__global__ void transposeCoarseGrained(float* odata,
	float* idata, int width, int height, int nreps)
{
	__shared__ float block[TILE_DIM][TILE_DIM + 1];
	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			block[threadIdx.y + i][threadIdx.x] =
				idata[index_in + i * width];
		}

		__syncthreads();
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index_out + i * height] =
				block[threadIdx.y + i][threadIdx.x];
		}
	}
}

__global__ void transposeDiagonal(float* odata,
	float* idata, int width, int height, int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int blockIdx_x, blockIdx_y;
	// diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	}
	else {
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}
	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;
	xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;
	for (int r = 0; r < nreps; r++) {
		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			tile[threadIdx.y + i][threadIdx.x] =
				idata[index_in + i * width];
		}

		__syncthreads();

		for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
			odata[index_out + i * height] =
				tile[threadIdx.x][threadIdx.y + i];
		}
	}
}
