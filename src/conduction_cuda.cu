/*! \file conduction_cuda.cu
*  \brief Function to calculate the thermal conduction between cells.*/

#ifdef CUDA
#ifdef CONDUCTION_GPU

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"conduction_cuda.h"

/*! \fn void calculate_heat_flux_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, 
                                      int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa)
 *  \brief Calculates the heat flux for the cells in the grid. */
__global__ void calculate_heat_flux_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma) {

  // Calculate the grid properties
  int n_cells = nx * ny * nz;
  int i_start, i_end, j_start, j_end, k_start, k_end;
  i_start = n_ghost;
  i_end = nx - n_ghost;
  if (ny == 1) {
    j_start = 0;
    j_end = 1;
  } else {
    j_start = n_ghost;
    j_end = ny-n_ghost;
  }
  if (nz == 1) {
    k_start = 0;
    k_end = 1;
  } else {
    k_start = n_ghost;
    k_end = nz-n_ghost;
  }

  Real right_flux, front_flux, up_flux;  // Each cell finds these, so in the end all fluxes are found

  // Calculate cell properties
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;

  // Find adjacent cell ids
  int right_id  = (xid + 1) + yid*nx + zid*nx*ny;
  int front_id  = xid + (yid + 1)*nx + zid*nx*ny;
  int up_id     = xid + yid*nx + (zid + 1)*nx*ny;

  
  __syncthreads();

  // Determine if the current cell should find the boundary fluxes
  bool validCell = xid >= i_start - 1 && yid >= j_start - 1 && zid >= k_start - 1 && xid < i_end && yid < j_end && zid < k_end;

  if(validCell) {

    // Find current cell temperature
    dev_flux_array[n_cells + id] = calculateTemp(dev_conserved, id, n_cells, gamma);

    // Calculate right boundary flux
    right_flux = calculateFlux(dev_conserved, dev_flux_array[n_cells + id], id, dev_flux_array[n_cells + right_id], right_id, n_cells, gamma, dx);
    // Store flux in global memory
    dev_flux_array[id] = right_flux;

    // Do y dimension if necessary
    if(ny > 1) {
      front_flux = calculateFlux(dev_conserved, dev_flux_array[n_cells + id], id, dev_flux_array[n_cells + front_id], front_id, n_cells, gamma, dy);
      dev_flux_array[n_cells + id] = front_flux;
    }

    // Do z dimension if neccessary
    if(nz > 1) {
      up_flux = calculateFlux(dev_conserved, dev_flux_array[n_cells + id], id, dev_flux_array[n_cells + up_id], up_id, n_cells, gamma, dz);
      dev_flux_array[2*n_cells + id] = up_flux;
    }
  }
}

/*! \fn void apply_heat_fluxes_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, 
                                      int n_ghost, Real dt, Real dx, Real dy, Real dz, Real *dt_array)
 *  \brief Apply the heat fluxes calculated in the previous kernel.  */
__global__ void apply_heat_fluxes_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, int n_ghost, Real dt, Real dx, Real dy, Real dz, Real *dt_array) {

  __shared__ Real min_dt[TPB];

  // Calculate grid properties
  int n_cells = nx * ny * nz;
  int i_start, i_end, j_start, j_end, k_start, k_end;
  i_start = n_ghost;
  i_end = nx - n_ghost;
  if (ny == 1) {
    j_start = 0;
    j_end = 1;
  } else {
    j_start = n_ghost;
    j_end = ny-n_ghost;
  }
  if (nz == 1) {
    k_start = 0;
    k_end = 1;
  } else {
    k_start = n_ghost;
    k_end = nz-n_ghost;
  }

  // Calculate cell properties
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;
  int tid = threadIdx.x;

  // Find adjacent cell ids (Opposite side of what was found in previous kernel)
  int left_id   = (xid - 1) + yid*nx + zid*nx*ny;
  int back_id   = xid + (yid - 1)*nx + zid*nx*ny;
  int down_id   = xid + yid*nx + (zid - 1)*nx*ny;

  // set min dt to a high number
  min_dt[tid] = 1e10;
  __syncthreads();

  // Should the current cell apply the fluxes
  bool validCell = xid >= i_start && yid >= j_start && zid >= k_start && xid < i_end && yid < j_end && zid < k_end;

  if(validCell) {
    // X
    Real right_flux = dev_flux_array[id];               // The previous kernel stored the right boundary flux at the current id
    Real left_flux = dev_flux_array[left_id];
    dev_conserved[4*n_cells + id] += (left_flux - right_flux)*(dt/dx);

    // Y
    if(ny > 1) {
      Real front_flux = dev_flux_array[n_cells + id];
      Real back_flux = dev_flux_array[n_cells + back_id];
      dev_conserved[4*n_cells + id] += (back_flux - front_flux)*(dt/dy);
    }

    // Z
    if(nz > 1) {
      Real up_flux = dev_flux_array[2*n_cells + id];
      Real down_flux = dev_flux_array[2*n_cells + down_id];
      dev_conserved[4*n_cells + id] += (down_flux - up_flux)*(dt/dz);
    }

    // Check the cell hasn't crashed
    if (dev_conserved[4*n_cells + id] > 0) {
      Real qa = dx*dx * dev_conserved[id] / kappa(dev_flux_array[n_cells + id]);
      Real min_dt_temp = qa / 4.0;
      if(ny > 1) min_dt_temp = qa / 8.0;
      if(nz > 1) min_dt_temp = qa / 6.0;
      min_dt[tid] = min_dt_temp;
    }
  }
  __syncthreads();

  // do the reduction in shared memory (find the min timestep in the block)
  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s) == 0) {
      min_dt[tid] = fmin(min_dt[tid], min_dt[tid + s]);
    }
    __syncthreads();
  }

  // write the result for this block to global memory
  if (tid == 0 && dt_array[blockIdx.x] > min_dt[0]) {
    dt_array[blockIdx.x] = min_dt[0];
  }
}

/*! \fn void calculateTemp(Real *dev_conserved, int id, int n_cells, Real gamma)
 *  \brief Calculate the temperature of the cell with the given id.  */
__device__ Real calculateTemp(Real *dev_conserved, int id, int n_cells, Real gamma) {
  Real mu = 1.0;
  
  Real d  =  dev_conserved[            id];        // Density
  Real E  =  dev_conserved[4*n_cells + id];        // Energy
  Real vx =  dev_conserved[1*n_cells + id] / d;    // Velocity X
  Real vy =  dev_conserved[2*n_cells + id] / d;    // Velocity Y
  Real vz =  dev_conserved[3*n_cells + id] / d;    // Velocity Z
  Real p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0); // Pressure
  p  = fmax(p, (Real) TINY_NUMBER);                // Make sure pressure isn't too low.

  Real T = gamma * p / d;

  return T; // Return temperature
}

/*! \fn void calculateFlux(Real *dev_conserved, Real cell_temp, int id_1, int id_2, 
                            int n_cells, Real gamma, Real kappa, Real del)
 *  \brief Calculate the flux between the two passed cells. The cell_temp is 
          also passed so the temperature of the current cell doesn't need to be
          calculated again for every boundary. */
__device__ Real calculateFlux(Real *dev_conserved, Real temp_1, int id_1, Real temp_2, int id_2, int n_cells, Real gamma, Real del) {
  Real d_avg = 0.5 * (dev_conserved[id_1] + dev_conserved[id_2]);
  Real T_avg = 0.5 * (temp_1 + temp_2);
  Real kd = kappa(T_avg) * d_avg;
  return kd*(temp_1 - temp_2)/del;
}

/*! \fn Real kappa(Real temp)
 *  \brief Calculate kappa given the passed temperature.  */
__device__ Real kappa(Real temp) {
  Real kappa = 0.00079218835705;
  return kappa;
}

#endif // CONDUCTION_GPU
#endif // CUDA
