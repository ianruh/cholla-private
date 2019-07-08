/*! \file conduction_cuda.cu
 *  \brief Function to calculate the thermal conduction between cells.*/

#ifdef CUDA
#ifdef CONDUCTION_GPU

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"conduction_cuda.h"

/*! \fn void conduction_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the energy
        of each cell according to thermal conduction. */
__global__ void conduction_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa) {

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

  Real right_flux, left_flux, front_flux, back_flux, up_flux, down_flux;

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
  int left_id   = (xid - 1) + yid*nx + zid*nx*ny;
  int front_id  = xid + (yid + 1)*nx + zid*nx*ny;
  int back_id   = xid + (yid - 1)*nx + zid*nx*ny;
  int up_id     = xid + yid*nx + (zid + 1)*nx*ny;
  int down_id   = xid + yid*nx + (zid - 1)*nx*ny;

  bool validCell = xid >= i_start && yid >= j_start && zid >= k_start && xid < i_end && yid < j_end && zid < k_end;

  if(validCell) {
    
    // Always do x dimension
    right_flux = calculateFlux(dev_conserved, id, right_id, n_cells, gamma, kappa); // Calc right bound flux
    left_flux = calculateFlux(dev_conserved, left_id, id, n_cells, gamma, kappa);   // Calc left bound flux
    dev_conserved[4*n_cells + id] += (left_flux - right_flux)*(dt/dx);              // Change energy

    // Do y dimension if necessary
    if(ny > 1) {
      front_flux = calculateFlux(dev_conserved, id, front_id, n_cells, gamma, kappa);
      back_flux = calculateFlux(dev_conserved, back_id, id, n_cells, gamma, kappa);
      dev_conserved[4*n_cells + id] += (back_flux - front_flux)*(dt/dy);
    }

    // Do z dimension if neccessary
    if(nz > 1) {
      up_flux = calculateFlux(dev_conserved, id, up_id, n_cells, gamma, kappa);
      down_flux = calculateFlux(dev_conserved, down_id, id, n_cells, gamma, kappa);
      // Update with z flux
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

__device__ Real calculateFlux(Real *dev_conserved, int id_1, int id_2, int n_cells, Real gamma, Real kappa) {
  Real temp_1 = calculateTemp(dev_conserved, id_1, n_cells, gamma);
  Real temp_2 = calculateTemp(dev_conserved, id_2, n_cells, gamma);

  Real kd = kappa * 0.5 * (dev_conserved[id_1] + dev_conserved[id_2]);
  return kd*(temp_1 - temp_2);
}

#endif // CONDUCTION_GPU
#endif // CUDA
