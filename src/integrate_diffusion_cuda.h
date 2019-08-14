#ifdef CONDUCTION_GPU

#include"grid3D.h"

#ifdef CONDUCTION_STS
void calc_STS_coeffs(Grid3D *G);
int get_N_STS(Real dt_MHD, Real dt_Diff);
void copy_constant_STS_coeffs(Grid3D *G);
void allocate_diffusion_memory(int nx, int ny, int nz);
#endif

#endif /* CONDUCTION_GPU */