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

__device__
float calculate_dE(const float *s_spins, const float trial, const int row, const int col,
                   const float step, const int site, const int sharedsite){
        float c = s_spins[sharedsite];
        float u = s_spins[(row - 1) * blockDim.x + col];
        float l = s_spins[row * blockDim.x + (col - 1)];
        float r = s_spins[row * blockDim.x + (col + 1)];
        float d = s_spins[(row + 1) * blockDim.x + col];
        float E0 = -(cosf(c-u)+cosf(c-l)+cosf(c-r)+cosf(c-d));
        float c1 = c + trial;
        float E1 = -(cosf(c1-u)+cosf(c1-l)+cosf(c1-r)+cosf(c1-d));
        return E1-E0;
    }


__device__
void chessBoardUpdate(float *s_spins, float *d_spins, float *d_random,
                      float *d_random_step, const float T, const int site, 
                      const int sharedsite, const float step){
    
    int row = threadIdx.y, col = threadIdx.x; 
    //Load spins to shared memory
    s_spins[sharedsite] = d_spins[site];
    __syncthreads();//wait until spins are loaded
    
    if (row == 0 || col == 0 || row == blockDim.y-1 || col == blockDim.x-1)
        return; //Edge site for shared memory filling
    
    int chess = (row%2 + col%2)%2;
    
    if (chess == 0){
        float trial = s_spins[sharedsite] + (d_random_step[site]-0.5)*2*M_PI*step;
        float dE = calculate_dE(s_spins, trial, row, col, 
                                step, site, sharedsite);
        if (exp(-dE/T) > d_random[site])
            s_spins[sharedsite] = trial;
    }
    
    __syncthreads();
    if (chess == 1){
        float trial = s_spins[sharedsite] + (d_random_step[site]-0.5)*2*M_PI*step;
        float dE = calculate_dE(s_spins, trial, row, col, 
                                step, site, sharedsite);
        if (exp(-dE/T) > d_random[site])
            s_spins[sharedsite] = trial;
    } 
    __syncthreads();

    //Update spins
    d_spins[site] = fmodf(s_spins[sharedsite], 2.f*M_PI);
    __syncthreads();
    
    return;
}


__global__
void isingSample(float *d_spins, float *d_random, float *d_random_step,
                 const float T, const int L, const float step){
    int tidx = threadIdx.x, tidy = threadIdx.y;
    // bdim - 2 because we don't update boundary
    int x = (int)(tidx + blockIdx.x * (blockDim.x - 2)) - 1;
    int y = (int)(tidy + blockIdx.y * (blockDim.y - 2)) - 1;
    if (x > L || y > L)
        return; //Prevents overlap with incommensurate blocks
    
    int col = MOD(x, L);
    int row = MOD(y, L);
    int site = row * L + col, sharedsite = tidy * blockDim.x + tidx;
    extern __shared__ float s_spins[];//(blockDim+2)**2

    chessBoardUpdate(s_spins, d_spins, d_random, d_random_step, T,
                     site, sharedsite, step);
    
    return;
}



