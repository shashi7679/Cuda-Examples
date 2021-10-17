//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define N 512
#include <stdio.h>
#include<stdlib.h>

__global__ void check_dim() {
	printf("\n BlockIdx.x : %d   BlockIdx.y : %d   BlockIdx.z : %d",blockIdx.x,blockIdx.y,blockIdx.z);
}
__global__ void Array_Initialization(int* arr) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	arr[id] = id;
}

int main() {
	cudaEvent_t start, stop;
	clock_t cpuStart, cpuEnd;
	dim3 grid(N, 1, 1);
	dim3 block(N, 1, 1);
	int* arr;
	double time_spent = 0.0;
	arr = (int*)malloc(sizeof(int) * N * N);
	cpuStart = clock();
	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			arr[i*N + j] = i * N + j;
		}
	}
	cpuEnd = clock();
	time_spent = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
	printf("Time taken in CPU initalization : %lf \n",time_spent);
	int* d;
	cudaMalloc(&d, sizeof(int) * N * N);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	Array_Initialization << <grid, block >> > (d);
	cudaMemcpy(arr, d, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float milli = 0.0;
	cudaEventElapsedTime(&milli, start, stop);
	printf("Time taken in GPU initalization : %f \n", milli);
	/*for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			printf(" %d ", arr[j + i * N]);
		}
		printf("\n");
	}*/
	//check_dim << <block,1 >> > ();
	return 0;
}