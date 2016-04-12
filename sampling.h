#ifndef __SAMPLING_H__
#define __SAMPLING_H__

__global__
void isingSample(float *d_spins, float *d_mx, float *d_my, float *d_localE, 
                 const float *d_random, const float *d_random_step, 
                 const float T, const int L, const float step);

#endif
