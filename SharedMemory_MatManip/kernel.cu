//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define N 32
#include <stdio.h>
#include<stdlib.h>

/*__global__ void check_dim() {
	printf("\n BlockIdx.x : %d   BlockIdx.y : %d   BlockIdx.z : %d",blockIdx.x,blockIdx.y,blockIdx.z);
}*/

__global__ void Array_Initialization(int* arr) {
	int id = N*threadIdx.y + threadIdx.x + N*N*blockIdx.x + N*N*N*blockIdx.y;
	arr[id] = id;
	//printf("\n id : %d    arr[id] = %d ", id, arr[id]);
}
__global__ void Matx_Manipulate(int* arr) {
	__shared__ int s[N*N];
	int id = N * threadIdx.y + threadIdx.x + N * N * blockIdx.x + N * N * N * blockIdx.y;
	int id_perblock = threadIdx.x + N * threadIdx.y;
	if (id_perblock == 1023) s[id_perblock] = 1024 * (blockIdx.x + blockIdx.y * N + 1);
	else s[id_perblock] = arr[id + 1];
	__syncthreads();
	arr[id] = s[id_perblock];
	//printf("\n id : %d    arr[id] = %d ", id, arr[id]);
}
int main() {
	cudaEvent_t start, stop;
	//clock_t cpuStart, cpuEnd;
	dim3 grid(N, N, 1);
	dim3 block(N, N, 1);
	int* arr;
	int* arr_mod;
	double time_spent = 0.0;
	arr = (int*)malloc(sizeof(int) * N * N* N * N);
	arr_mod = (int*)malloc(sizeof(int) * N * N * N * N);
	//cpuStart = clock();
	/*for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			arr[i*N + j] = i * N + j;
		}
	}
	cpuEnd = clock();
	time_spent = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;
	printf("Time taken in CPU initalization : %lf \n",time_spent);*/
	int* d;
	cudaMalloc(&d, sizeof(int) * N * N* N * N);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	Array_Initialization << <grid, block >> > (d);
	cudaMemcpy(arr, d, sizeof(int) * N * N * N * N, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	
	cudaEventSynchronize(stop);
	float milli = 0.0;
	cudaEventElapsedTime(&milli, start, stop);
	printf("Time taken in GPU initalization : %f \n", milli);
	printf("\n Before Modification :- ");
	for (int i = 0;i < N*N;i++) {
		for (int j = 0;j < N * N;j++) {
			printf(" %d ", arr[i*N*N + j]);
		}
		printf("\n");
	}
	Matx_Manipulate << <grid, block >> > (d);
	cudaMemcpy(arr_mod, d, sizeof(int) * N * N * N * N, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("\n After Modification :- ");
	for (int i = 0;i < N * N;i++) {
		for (int j = 0;j < N * N;j++) {
			printf(" %d ", arr_mod[i * N * N + j]);
		}
		printf("\n");
	}
	return 0;
}