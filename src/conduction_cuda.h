/*! \file conduction_cuda.h
 *  \brief Declarations of conduction kernel. */

#ifdef CUDA
#ifdef CONDUCTION_GPU

#include<cuda.h>
#include<math.h>
#include"global.h"


/*! \fn void conduction_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the energy
        of each cell according to thermal conduction. */
__global__ void conduction_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa);

__device__ Real calculateTemp(Real *dev_conserved, int id, Real gamma, Real kappa);

#endif // CONDUCTION_GPU
#endif // CUDA
