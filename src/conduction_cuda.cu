/*! \file conduction_cuda.cu
 *  \brief Function to calculate the thermal conduction between cells.*/

#ifdef CUDA
#ifdef CONDUCTION_GPU

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"conduction_cuda.h"

// Calculates the fluxes for each cell.
__global__ void calculate_heat_flux_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa) {

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

  Real right_flux, front_flux, up_flux;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;
  // and a thread id within the block
  // int tid = threadIdx.x;

  // FYI id = xid + yid*nx + zid*nx*ny
  int right_id  = (xid + 1) + yid*nx + zid*nx*ny;
  int front_id  = xid + (yid + 1)*nx + zid*nx*ny;
  int up_id     = xid + yid*nx + (zid + 1)*nx*ny;

  bool validCell = xid >= i_start - 1 && yid >= j_start - 1 && zid >= k_start - 1 && xid < i_end && yid < j_end && zid < k_end;

  if(validCell) {

    Real cellTemp = calculateTemp(dev_conserved, id, n_cells, gamma);
    
    // Always do x dimension
    right_flux = calculateFlux(dev_conserved, cellTemp, id, right_id, n_cells, gamma, kappa, dx); // Calc right bound flux
    dev_flux_array[id] = right_flux;

    // Do y dimension if necessary
    if(ny > 1) {
      front_flux = calculateFlux(dev_conserved, cellTemp, id, front_id, n_cells, gamma, kappa, dy);
      dev_flux_array[n_cells + id] = front_flux;
    }

    // Do z dimension if neccessary
    if(nz > 1) {
      up_flux = calculateFlux(dev_conserved, cellTemp, id, up_id, n_cells, gamma, kappa, dz);
      dev_flux_array[2*n_cells + id] = up_flux;
    }
  }
}

// Applies the previously calculated fluxes
__global__ void apply_heat_fluxes_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, int n_ghost, Real dt, Real dx, Real dy, Real dz) {

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

  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;

  // FYI id = xid + yid*nx + zid*nx*ny
  int left_id   = (xid - 1) + yid*nx + zid*nx*ny;
  int back_id   = xid + (yid - 1)*nx + zid*nx*ny;
  int down_id   = xid + yid*nx + (zid - 1)*nx*ny;

  bool validCell = xid >= i_start && yid >= j_start && zid >= k_start && xid < i_end && yid < j_end && zid < k_end;

  if(validCell) {
    // X
    Real right_flux = dev_flux_array[id];
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
  }
}

__device__ Real calculateTemp(Real *dev_conserved, int id, int n_cells, Real gamma) {
  Real d  =  dev_conserved[            id];        // Density
  Real E  =  dev_conserved[4*n_cells + id];        // Energy
  Real vx =  dev_conserved[1*n_cells + id] / d;    // Velocity X
  Real vy =  dev_conserved[2*n_cells + id] / d;    // Velocity Y
  Real vz =  dev_conserved[3*n_cells + id] / d;    // Velocity Z
  Real p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0); // Pressure
  p  = fmax(p, (Real) TINY_NUMBER);                // Make sure pressure isn't too low.
  return p / d; // Return temperature
}

__device__ Real calculateFlux(Real *dev_conserved, Real cell_temp, int id_1, int id_2, int n_cells, Real gamma, Real kappa, Real del) {
  Real temp_2 = calculateTemp(dev_conserved, id_2, n_cells, gamma);

  Real kd = kappa * 0.5 * (dev_conserved[id_1] + dev_conserved[id_2]);
  return kd*(cell_temp - temp_2)/del;
}

#endif // CONDUCTION_GPU
#endif // CUDA
