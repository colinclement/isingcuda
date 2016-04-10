#ifndef __SAMPLING_H__
#define __SAMPLING_H__

__global__
void isingSample(float *d_spins, float *d_random, float *d_random_step,
                 const float T, const int L, const float step);

__device__
void chessBoardUpdate(float *s_spins, float *d_spins, float *d_random,
                      float *d_random_step, const float T, const int site, 
                      const int sharedsite, const float step );

#endif
