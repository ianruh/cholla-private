#ifdef CUDA

#include "global.h"

/* \fn void Copy_Custom_Params(Real* parameters)
* \brief Copy the passed parameters to constant memory on the GPU */
void Copy_Custom_Params(Real* parameters);

void Copy_Temp_Init(Real temp);

#endif