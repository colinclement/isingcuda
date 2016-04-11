#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <math.h>

#define MOD(x, N) (((x < 0) ? ((x % N) + N) : x) % N)
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif

#define PI_F 3.141592654f
#define TPI_F 6.283185307f

__device__
float calculate_dE(const float *s_spins, const float trial,
                   const float step, const int sharedsite){
        int tidy = threadIdx.y, tidx = threadIdx.x; 
        
        float c = s_spins[sharedsite];
        float u = s_spins[(tidy - 1) * blockDim.x + tidx];
        float l = s_spins[tidy * blockDim.x + (tidx - 1)];
        float r = s_spins[tidy * blockDim.x + (tidx + 1)];
        float d = s_spins[(tidy + 1) * blockDim.x + tidx];
        
        float E0 = -(cosf(c -u)+cosf(c -l)+cosf(c -r)+cosf(c -d));
        float E1 = -(cosf(trial-u)+cosf(trial-l)+
                     cosf(trial-r)+cosf(trial-d));
        return E1-E0;
    }

__global__
void isingSample(float *d_spins, const float *d_random, const float *d_random_step, 
                 const float T, const int L, const float step){
    
    int tidx = threadIdx.x, tidy = threadIdx.y;
    // bdim - 2 because we don't update boundary
    int x = (int)(tidx + blockIdx.x * (blockDim.x - 2)) - 1;
    int y = (int)(tidy + blockIdx.y * (blockDim.y - 2)) - 1;
    if (x > L || y > L)
        return; //Limits to system + nearest neighbor
    
    int col = MOD(x, L), row = MOD(y, L);
    int site = row*L + col, sharedsite = tidy * blockDim.x + tidx;
    extern __shared__ float s_spins[];

    s_spins[sharedsite] = d_spins[site];
    __syncthreads();//wait until spins are loaded
   
    int edge = tidx==0 || tidy==0 || tidy==blockDim.y-1 || tidx==blockDim.x-1;
    if (edge || x == L || y == L)
        return; //Edge sites don't get updated! 

    int chess = (tidx%2 + tidy%2)%2;
    
    if (chess == 0){
        float trial = s_spins[sharedsite] + (d_random_step[site]-0.5f)*TPI_F*step;
        float dE = calculate_dE(s_spins, trial, step, sharedsite);
        if (dE < 0 || expf(-dE/T) > d_random[site])
            s_spins[sharedsite] = trial;
    }
    __syncthreads();
    if (chess == 1){
        float trial = s_spins[sharedsite] + (d_random_step[site]-0.5f)*TPI_F*step;
        float dE = calculate_dE(s_spins, trial, step, sharedsite);
        if (dE < 0 || expf(-dE/T) > d_random[site])
            s_spins[sharedsite] = trial;
    } 
    __syncthreads();

    //Update spins
    d_spins[site] = s_spins[sharedsite];
    return;
}


