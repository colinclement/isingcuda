#ifndef __SAMPLING_H__
#define __SAMPLING_H__

__global__
void isingSample(int *d_spins, float *d_random, const float T,
                 const int L);

__device__
void chessBoardUpdate(int *s_spins, int *d_spins, float *d_random, 
                      const float T, const int site, const int sharedsite);

#endif
