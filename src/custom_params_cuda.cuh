#ifdef CUDA

#include "global.h"

// Constant array to hold up to 100 user-defined parameters
extern __constant__ Real custom_params[100];

__global__ void testFunc();

#endif