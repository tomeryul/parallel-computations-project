#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void build_ReturndArr(double* data, double* matValues,int arrSize, double t) {

	int index = threadIdx.x;
	if(index < arrSize){
		double X = (data[index*5+2]-data[index*5+1])/2 * sin(t*M_PI/2) + (data[index*5+2]+data[index*5+1])/2;
		double Y = data[index*5+3]*X+data[index*5+4];
	
		matValues[index] = X;
		matValues[index+arrSize] = Y;
	
		
	}
}


double* GPUGetXY(double *data, int numElements, double t){

	cudaError_t err = cudaSuccess;
	
	// Allocate memory on GPU and CPU for the returnd Arr of X and Y
	size_t size_ReturndArrXY = (numElements/5)*2 * sizeof(double);
	double* h_myArr = (double*)malloc(size_ReturndArrXY);
	
	double *d_myArr = NULL;
    	err = cudaMalloc((void **)&d_myArr, size_ReturndArrXY);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to allocate returndArrXY memory - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
 
    	// Allocate memory on GPU to copy the data from the host
    	double *d_ArrData = NULL;
    	size_t size = numElements * sizeof(double);
    	err = cudaMalloc((void **)&d_ArrData, size);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to allocate device memory for d_ArrData- %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy data from host to the GPU memory
    	err = cudaMemcpy(d_ArrData, data, size, cudaMemcpyHostToDevice);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	 }
   	 
   	  // Launch the Kernel
    	build_ReturndArr <<< 1 , PART >>> (d_ArrData, d_myArr, numElements/5, t);
    	
    	// Copy the  result of the d_myArr from GPU to the host memory.
    	err = cudaMemcpy(h_myArr, d_myArr, size_ReturndArrXY, cudaMemcpyDeviceToHost);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy d_myArr to host - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
    	  err = cudaGetLastError();
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to launch kernel function -  %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
    	 // Free allocated memory on GPU
    	if (cudaFree(d_myArr) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
     	// Free allocated memory on GPU
    	if (cudaFree(d_ArrData) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	
	
	return h_myArr; 
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void init_HelpArr(int* helpArr) {

	int index = threadIdx.x;
	
	helpArr[index] = -1;
	
}

__global__ void build_HelpArr(int* d_helpArr, double* d_ArrValuesX, double* d_ArrValuesY, int sizeArr, double D, int K, int StartIndex) {

	
	int numClous = 0;
	double distanse;
	int index = threadIdx.x;
		
	int i = StartIndex*PART + index;


		
	if(i < sizeArr){
		for(int j = 0 ; j < sizeArr ; j++){
			if(i != j){
				distanse = sqrt(((d_ArrValuesX[j]-d_ArrValuesX[i])*(d_ArrValuesX[j]-d_ArrValuesX[i]))+((d_ArrValuesY[j]-d_ArrValuesY[i])*(d_ArrValuesY[j]-d_ArrValuesY[i])));
				
				if(distanse < D){
					numClous+=1;
					if(numClous == K){
						d_helpArr[index] = i;
						break;
					}
				}
			
			}
		}
	}
}


int* GPUGetPoints(double *allValuesX, double *allValuesY, int sizeArr, double D, int K, int StartIndex){

	cudaError_t err = cudaSuccess;
	int numPointsReturn = 0;

	
	int* h_returndPointsArr = (int*)malloc(3*sizeof(int));
	h_returndPointsArr[0] = -1;
	h_returndPointsArr[1] = -1;
	h_returndPointsArr[2] = -1;
	
	// Allocate memory on GPU and CPU for the helpArr
	// every point will have 1 spaces on the arr for 1 points thet is clous
	size_t size_helpArr = (PART) * sizeof(int);
	int* h_helpArr = (int*)malloc(size_helpArr);
	
	int *d_helpArr = NULL;
    	err = cudaMalloc((void **)&d_helpArr, size_helpArr);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to allocate device memory for d_helpArr - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
    	// Allocate memory on GPU to copy the allValuesX from the host
    	double *d_ArrValuesX = NULL;
    	size_t theSize = sizeArr * sizeof(double);
    	err = cudaMalloc((void **)&d_ArrValuesX, theSize);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to allocate device memory for d_ArrValuesX- %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy data from host to the GPU memory
    	err = cudaMemcpy(d_ArrValuesX, allValuesX, theSize, cudaMemcpyHostToDevice);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	 }
    	
    	// Allocate memory on GPU to copy the allValuesY from the host
    	double *d_ArrValuesY = NULL;
    	err = cudaMalloc((void **)&d_ArrValuesY, theSize);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to allocate device memory for d_ArrValuesY- %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	// Copy data from host to the GPU memory
    	err = cudaMemcpy(d_ArrValuesY, allValuesY, theSize, cudaMemcpyHostToDevice);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
   	 }
   	 
   	// Launch the Kernel
    	init_HelpArr <<< 1 , PART >>> (d_helpArr);
   	build_HelpArr <<< 1 , PART >>> (d_helpArr, d_ArrValuesX, d_ArrValuesY, sizeArr, D, K, StartIndex);
   	
   	
   	// Copy the  result of the d_helpArr from GPU to the host memory.
    	err = cudaMemcpy(h_helpArr, d_helpArr, size_helpArr, cudaMemcpyDeviceToHost);
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to copy d_helpArr to host - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
    	
    	for(int i = 0 ; i < PART ; i++){
    		if(h_helpArr[i]!= -1){
    			h_returndPointsArr[numPointsReturn] = h_helpArr[i];
    			numPointsReturn += 1;
    		}
    		if(numPointsReturn == 3)
    			break;
    	}
    	
    	err = cudaGetLastError();
    	if (err != cudaSuccess) {
        	fprintf(stderr, "Failed to launch kernel function -  %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
    	 // Free allocated memory on GPU
    	if (cudaFree(d_helpArr) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
     	// Free allocated memory on GPU
    	if (cudaFree(d_ArrValuesX) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	
    	// Free allocated memory on GPU
    	if (cudaFree(d_ArrValuesY) != cudaSuccess) {
        	fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}
    	

   	
	return h_returndPointsArr;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
