#ifdef CUDA

// #include "global.h"
#include <stdio.h>
#include "global.h"
#include "custom_params_cuda.cuh"
#include "custom_params_cuda.h"

// Define the array of params in constant memory (declared in the .cuh)
__constant__ Real custom_params[100];

/* \fn void Copy_Custom_Params(Real* parameters)
* \brief Copy the passed parameters to constant memory on the GPU */
void Copy_Custom_Params(Real* parameters) {
    printf("Custom Parameters: [%d, %d, %d, %d, %d, ...]\n", parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]);
    cudaMemcpyToSymbol(custom_params, parameters, 100*sizeof(Real));
    testFunc<<<1,1>>>();
}

__global__ void testFunc() {
    printf("Custom Parameter GPU: %f\n", custom_params[0]);
}

#endif