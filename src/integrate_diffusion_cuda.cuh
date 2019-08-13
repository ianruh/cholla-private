#ifdef CUDA
#ifdef CONDUCTION_STS

extern __constant__ Real mu[101];
extern __constant__ Real nu[101];
extern __constant__ Real ajm1[101];

extern __device__ Real *Y0;
extern __device__ Real *Lclass0;
extern __device__ Real *Yjm2;

#endif /* CONDUCTION_STS */
#endif /* CUDA */