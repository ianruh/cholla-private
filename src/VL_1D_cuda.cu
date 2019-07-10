/*! \file VL_1D_cuda.cu
 *  \brief Definitions of the cuda VL algorithm functions. */

#ifdef CUDA
#ifdef VL

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"VL_1D_cuda.h"
#include"pcm_cuda.h"
#include"plmp_cuda.h"
#include"plmc_cuda.h"
#include"ppmp_cuda.h"
#include"ppmc_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"cooling_cuda.h"
#include"conduction_cuda.h"
#include"error_handling.h"
#include"io.h"


__global__ void Update_Conserved_Variables_1D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F, 
                                                   int n_cells, int n_ghost, Real dx, Real dt, Real gamma, int n_fields);



Real VL_Algorithm_1D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt, int n_fields)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //host_conserved0 contains the values at time n
  //host_conserved1 will contain the values at time n+1

  // Initialize dt values
  Real max_dti = 0;
  #ifdef COOLING_GPU
  Real min_dt = 1e10;
  #endif  

  int n_cells = nx;
  int ny = 1;
  int nz = 1;

  // set the dimensions of the cuda grid
  ngrid = (n_cells + TPB - 1) / TPB;
  dim3 dimGrid(ngrid, 1, 1);
  dim3 dimBlock(TPB, 1, 1);

  if ( !memory_allocated ) {

    // allocate an array on the CPU to hold max_dti returned from each thread block
    host_dti_array = (Real *) malloc(ngrid*sizeof(Real));
    #ifdef COOLING_GPU
    host_dt_array = (Real *) malloc(ngrid*sizeof(Real));
    #endif
  
    // allocate memory on the GPU
    CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_conserved_half, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,   n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif  
    #ifdef CONDUCTION_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_flux_array, nx*sizeof(Real)) );
    #endif


    #ifndef DYNAMIC_GPU_ALLOC 
    // If memory is single allocated: memory_allocated becomes true and succesive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif 
  }

  // copy the conserved variable array onto the GPU
  CudaSafeCall( cudaMemcpy(dev_conserved, host_conserved0, n_fields*n_cells*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaCheckError();

  // Step 1: Use PCM reconstruction to put conserved variables into interface arrays
  PCM_Reconstruction_1D<<<dimGrid,dimBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx, n_ghost, gama, n_fields);
  CudaCheckError();

  // Step 2: Calculate first-order upwind fluxes 
  #ifdef EXACT
  Calculate_Exact_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  CudaCheckError();


  // Step 3: Update the conserved variables half a timestep 
  Update_Conserved_Variables_1D_half<<<dimGrid,dimBlock>>>(dev_conserved, dev_conserved_half, F_x, n_cells, n_ghost, dx, 0.5*dt, gama, n_fields);
  CudaCheckError();


  // Step 4: Construct left and right interface values using updated conserved variables
  #ifdef PCM
  PCM_Reconstruction_1D<<<dimGrid,dimBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx, n_ghost, gama, n_fields);
  #endif
  #ifdef PLMC
  PLMC_cuda<<<dimGrid,dimBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  #endif  
  #ifdef PLMP
  PLMP_cuda<<<dimGrid,dimBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  #endif
  #ifdef PPMP
  PPMP_cuda<<<dimGrid,dimBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  #endif
  #ifdef PPMC
  PPMC_cuda<<<dimGrid,dimBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  #endif
  CudaCheckError();


  // Step 5: Calculate the fluxes again
  #ifdef EXACT
  Calculate_Exact_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  CudaCheckError();


  // Step 6: Update the conserved variable array
  Update_Conserved_Variables_1D<<<dimGrid,dimBlock>>>(dev_conserved, F_x, n_cells, x_off, n_ghost, dx, xbound, dt, gama, n_fields);
  CudaCheckError();
   

  #ifdef DE
  Sync_Energies_1D<<<dimGrid,dimBlock>>>(dev_conserved, nx, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif    


  // Apply cooling
  #ifdef COOLING_GPU
  cooling_kernel<<<dimGrid,dimBlock>>>(dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gama, dev_dt_array);
  CudaCheckError();
  #endif

  // Thermal Conduction
  #ifdef CONDUCTION_GPU
  Real kappa = 1.0;
  calculate_heat_flux_kernel<<<dimGrid, dimBlock>>>(dev_conserved, dev_flux_array, nx, ny, nz, n_ghost, n_fields, dt, dx, 1, 1, gama, kappa);
  cudaError_t err = cudaGetLastError();
  gpuErrchk(err);
  CudaCheckError();
  cudaDeviceSynchronize();
  apply_heat_fluxes_kernel<<<dimGrid, dimBlock>>>(dev_conserved, dev_flux_array, nx, ny, nz, n_ghost, dt, dx, 1, 1);
  err = cudaGetLastError();
  gpuErrchk(err);
  CudaCheckError();
  #endif

  
  // Step 7: Calculate the next timestep
  Calc_dt_1D<<<dimGrid,dimBlock>>>(dev_conserved, n_cells, n_ghost, dx, dev_dti_array, gama);
  CudaCheckError();


  // copy the conserved variable array back to the CPU
  CudaSafeCall( cudaMemcpy(host_conserved1, dev_conserved, n_fields*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );

  // copy the dti array onto the CPU
  CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
  // iterate through to find the maximum inverse dt for this subgrid block
  for (int i=0; i<ngrid; i++) {
    max_dti = fmax(max_dti, host_dti_array[i]);
  }
  #ifdef COOLING_GPU
  // copy the dt array from cooling onto the CPU
  CudaSafeCall( cudaMemcpy(host_dt_array, dev_dt_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
  // find maximum inverse timestep from cooling time
  for (int i=0; i<ngrid; i++) {
    min_dt = fmin(min_dt, host_dt_array[i]);
  }  
  if (min_dt < C_cfl/max_dti) {
    max_dti = C_cfl/min_dt;
  }
  #endif

  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_VL_1D();
  #endif


  // return the maximum inverse timestep
  return max_dti;


}

void Free_Memory_VL_1D() {

  // free the CPU memory
  free(host_dti_array);
  #ifdef COOLING_GPU
  free(host_dt_array);  
  #endif  

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(F_x);
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  cudaFree(dev_dt_array);
  #endif

}

__global__ void Update_Conserved_Variables_1D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F, int n_cells, int n_ghost, Real dx, Real dt, Real gamma, int n_fields)
{
  int id, imo;
  Real dtodx = dt/dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  
  #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, P;
  int ipo;
  #endif

  // threads corresponding all cells except outer ring of ghost cells do the calculation
  if (id > 0 && id < n_cells-1)
  {
    imo = id-1;
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (d < 0.0 || d != d) printf("Negative density before half step update.\n");
    //if (P < 0.0) printf("%d Negative pressure before half step update.\n", id);
    ipo = id+1;
    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo]; 
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo]; 
    #endif
    // update the conserved variable array
    dev_conserved_half[            id] = dev_conserved[            id] + dtodx * (dev_F[            imo] - dev_F[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id] + dtodx * (dev_F[  n_cells + imo] - dev_F[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id] + dtodx * (dev_F[2*n_cells + imo] - dev_F[2*n_cells + id]);
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id] + dtodx * (dev_F[3*n_cells + imo] - dev_F[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id] + dtodx * (dev_F[4*n_cells + imo] - dev_F[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved_half[(5+i)*n_cells + id] = dev_conserved[(5+i)*n_cells + id] + dtodx * (dev_F[(5+i)*n_cells + imo] - dev_F[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved_half[(n_fields-1)*n_cells + id] = dev_conserved[(n_fields-1)*n_cells + id] 
                                       + dtodx * (dev_F[(n_fields-1)*n_cells + imo] - dev_F[(n_fields-1)*n_cells + id])
                                       + 0.5*P*(dtodx*(vx_imo-vx_ipo));
    #endif    
  }


}





#endif //VL
#endif //CUDA
