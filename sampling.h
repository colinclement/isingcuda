#ifndef __SAMPLING_H__
#define __SAMPLING_H__

__global__
void isingSample(float *d_spins, const float *d_random, const float *d_random_step,
                 const float T, const int L, const float step);

#endif
