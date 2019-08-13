#ifdef CUDA
#ifdef CONDUCTION_GPU
#ifdef CONDUCTION_STS


#include "global.h"
#include "integrate_diffusion_cuda.h"
#include "integrate_diffusion_cuda.cuh"
#include "global_cuda.h"

__constant__ Real mu[101];
__constant__ Real nu[101];
__constant__ Real ajm1[101];
__device__ Real *Y0;
__device__ Real *Lclass0;
__device__ Real *Yjm2;

void calc_STS_coeffs(Grid3D *G) {
    int j, nl, nd;
    Real jval, jinv;
    Real TWO_3RDS = 2.0/3.0;
    Real ONE_3RD = 1.0/3.0;

    Real a[N_STS_MAX + 1], b[N_STS_MAX + 1];

    a[0] = TWO_3RDS;
    a[1] = TWO_3RDS;
    b[0] = ONE_3RD;
    b[1] = ONE_3RD;

    for (j = 2; j <= N_STS_MAX; j++) {
        jval = (Real)(j);
        jinv = 1.0 / jval;
        b[j] = 0.5 - jinv + 1.0 / (1.0 + jval);
        a[j] = 1.0 - b[j];
    }


    G->mu[0] = 0.0;
    G->mu[1] = 0.0;
    G->nu[0] = 0.0;
    G->nu[1] = 0.0;
    G->ajm1[0] = 0.0;
    for (j = 2; j <= N_STS_MAX; j++) {
        jval = (Real)(j);
        jinv = 1.0 / jval;
        G->mu[j] = (2.0 * jval - 1.0) * jinv * b[j] / b[j - 1];
        G->nu[j] = -(jval - 1.0) * jinv * b[j] / b[j - 2];
        G->ajm1[j] = a[j - 1];
    }

    for (j = 0; j <= N_STS_MAX; j++) {
        printf("j = %2d\t mu = %.4f\t nu = %.4f\t, 1-mu-nu = %.4f\n", j, G->mu[j], G->nu[j], 1.0 - G->mu[j] - G->nu[j]);
    }

    return;
}

// Calculate the number of substeps for STS
int get_N_STS(Real dt_MHD, Real dt_Diff) {
  int s = 1;
  Real dt_ratio = dt_MHD/dt_Diff;
  Real sval,criteria = 0.0;
  while (dt_ratio > criteria && s < N_STS_MAX) {
    s += 2;
    sval = (Real)(s);
    criteria = 0.25*(sval*(sval+1.0) - 2.0);
  }
  return s; // number of substeps (stages) in a super timestep 
}

// Copy coefficients over to constant memory
void copy_constant_STS_coeffs(Grid3D *G) {
    cudaMemcpyToSymbol(mu, G->mu, 101*sizeof(Real));
    cudaMemcpyToSymbol(nu, G->nu, 101*sizeof(Real));
    cudaMemcpyToSymbol(ajm1, G->ajm1, 101*sizeof(Real));
}

// Allocate memory for Y0, Lclass0, and Yjm2 in global memory
void allocate_diffusion_memory(int nx, int ny, int nz) {
    CudaSafeCall( cudaMalloc((void**)&Y0, nx*ny*nz*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Lclass0, nx*ny*nz*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Yjm2, nx*ny*nz*sizeof(Real)) );
}

#endif /* CONDUCTION_STS */
#endif /* CONDUCTION_GPU */
#endif /* CUDA */