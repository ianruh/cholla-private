/*! \file conduction_cuda.cu
 *  \brief Function to calculate the thermal conduction between cells.*/

#ifdef CUDA
#ifdef CONDUCTION_GPU

#include<cuda.h>
#include<math.h>
#include"global.h"
#include"global_cuda.h"
#include"conduction_cuda.h"

extern texture<float, 2, cudaReadModeElementType> coolTexObj;
extern texture<float, 2, cudaReadModeElementType> heatTexObj;

/*! \fn void conduction_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the energy
        of each cell according to thermal conduction. */
__global__ void conduction_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa) {

  // Only allocate as much shared memory as needed;
  int numDim = 1;
  numDim += (ny == 1) ? 0 : 1;
  numDim += (nz == 1) ? 0 : 1;

  __shared__ Real shared[TPB*(numDim + 1)];
  // shared[tid] = temp
  // shared[TPB + tid] = flux x
  // shared[2*TPB + tid] = flux y
  // shared[3*TPB + tid] = flux z

  int n_cells = nx * ny * nz;
  int i_start, i_end, j_start, j_end, k_start, k_end;
  i_start = n_ghost;
  i_end = nx - n_ghost;
  if (ny == 1) {
    j_start = 0;
    j_end = 1;

    // Allocate y dim flux array
    Qy = 
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
 
  Real d, E;                    // density, energy
  Real n, T, T_init;            // number density, temperature, initial temperature
  Real del_T, dt_sub;           // change in temp, sub-time step
  Real mu;                      // mean molecular weight
  Real cool;                    //cooling rate per volume, erg/s/cm^3
  Real vx, vy, vz, p;           // (x,y,z) velocity, pressure
  Real T_min = 1.0e4;           // minimum temperature allowed
  Reak kd;                      // kappa * density

  mu = 0.6;
  //mu = 1.27;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int id = threadIdx.x + blockId * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;
  // and a thread id within the block
  int tid = threadIdx.x;

  // FYI id = xid + yid*nx + zid*nx*ny
  int right_id = (xid + 1) + yid*nx + zid*nx*ny;
  int left_id = (xid - 1) + yid*nx + zid*nx*ny;
  int front_id = xid + (yid + 1)*nx + zid*nx*ny;
  int back_id = xid + (yid - 1)*nx + zid*nx*ny;
  int up_id = xid + yid*nx + (zid + 1)*nx*ny;
  int down_id = xid + yid*nx + (zid - 1)*nx*ny;

  int right_tid = right_id - blockId * blockDim.x;
  int left_tid = left_id - blockId * blockDim.x;
  int front_tid = front_id - blockId * blockDim.x;
  int back_tid = back_id - blockId * blockDim.x;
  int up_tid = up_id - blockId * blockDim.x;
  int down_tid = down_id - blockId * blockDim.x;

  __syncthreads();

  ///////////////// Calculate Cell Temperature ////////////////
  // Ghost cells need to do this to span the stencil
  d  =  dev_conserved[            id];        // Density
  E  =  dev_conserved[4*n_cells + id];        // Energy
  if (E < 0.0 || E != E) return;              // Make sure thread is alive
  vx =  dev_conserved[1*n_cells + id] / d;    // Velocity X
  vy =  dev_conserved[2*n_cells + id] / d;    // Velocity Y
  vz =  dev_conserved[3*n_cells + id] / d;    // Velocity Z
  p  = (E - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0); // Pressure
  p  = fmax(p, (Real) TINY_NUMBER);           // Make sure pressure isn't too low.

  // calculate the number density of the gas (in cgs)
  n = d*DENSITY_UNIT / (mu * MP);
  T_init = p*PRESSURE_UNIT/ (n*KB);
  shared[tid] = T_init;
  __syncthreads();
  
  // Calculate 1D Fluxes
  if(xid > i_start && xid < i_end + 1) {
    kd = kappa * 0.5 * (d + dev_conserved[left_id]);
    shared[TPB + tid] = kd*(shared[tid] - shared[left_tid]);
  }

  // Calculate 2D Fluxes
  if(ny != 1 && yid > j_start && yid < i_end + 1) {
    kd = kappa * 0.5 * (d + dev_conserved[back_id]);
    shared[2*TPB + tid] = kd*(shared[tid] - shared[back_tid]);
  }

  // Calculate 3D Fluxes
  if(nz != 1 && zid > k_start && zid < k_end + 1) {
    kd = kappa * 0.5 * (d + dev_conserved[down_id]);
    shared[3*TPB + tid] = kd*(shared[tid] - shared[down_tid]);
  }

  // only threads corresponding to real cells update energy
  if (xid >= i_start && xid < i_end && yid >= j_start && yid < j_end && zid >= k_start && zid < k_end) {
    
    // Update with x flux
    dev_conserved[4*n_cells + id] += (shared[TPB + right_tid] - shared[TPB + tid])*(dt/dx);

    // Update with y flux
    if(ny != 1) {
      dev_conserved[4*n_cells + id] += (shared[TPB + front_tid] - shared[TPB + tid])*(dt/dy);
    }

    // Update with z flux
    if(nz != 1) {
      dev_conserved[4*n_cells + id] += (shared[TPB + up_tid] - shared[TPB + tid])*(dt/dz);
    }
  }
  return;
}

#endif // CONDUCTION_GPU
#endif // CUDA