/*! \file conduction_cuda.h
 *  \brief Declarations of conduction kernel. */

#ifdef CUDA
#ifdef CONDUCTION_GPU

#include<cuda.h>
#include<math.h>
#include"global.h"


/*! \fn void calculate_heat_flux_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, 
                                      int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa)
 *  \brief Calculates the heat flux for the cells in the grid. */
__global__ void calculate_heat_flux_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real dx, Real dy, Real dz, Real gamma, Real kappa);

/*! \fn void apply_heat_fluxes_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, 
                                      int n_ghost, Real dt, Real dx, Real dy, Real dz)
 *  \brief Apply the heat fluxes calculated in the previous kernel.  */
__global__ void apply_heat_fluxes_kernel(Real *dev_conserved, Real *dev_flux_array, int nx, int ny, int nz, int n_ghost, Real dt, Real dx, Real dy, Real dz);

/*! \fn void calculateTemp(Real *dev_conserved, int id, int n_cells, Real gamma)
 *  \brief Calculate the temperature of the cell with the given id.  */
__device__ Real calculateTemp(Real *dev_conserved, int id, int n_cells, Real gamma);

/*! \fn void calculateFlux(Real *dev_conserved, Real cell_temp, int id_1, int id_2, 
                            int n_cells, Real gamma, Real kappa, Real del)
 *  \brief Calculate the flux between the two passed cells. The cell_temp is 
          also passed so the temperature of the current cell doesn't need to be
          calculated again for every boundary. */
__device__ Real calculateFlux(Real *dev_conserved, Real cell_temp, int id_1, int id_2, int n_cells, Real gamma, Real kappa, Real del);

#endif // CONDUCTION_GPU
#endif // CUDA
