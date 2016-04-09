#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>

#define MOD(x, N) (((x < 0) ? ((x % N) + N) : x) % N)
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif

__device__
void chessBoardUpdate(int *s_spins, int *d_spins, float *d_random, 
                      const float T, const int site, const int sharedsite){
    
    int row = threadIdx.y, col = threadIdx.x; 
    //Load spins to shared memory
    s_spins[sharedsite] = d_spins[site];
    __syncthreads();//wait until spins are loaded
    
    if (row == 0 || col == 0 || row == blockDim.y-1 || col == blockDim.x-1)
        return; //Edge site for shared memory filling
    
    int neighSum = 0;
    int chess = (row%2 + col%2)%2;
    int spin = s_spins[sharedsite];
    
    if (chess == 0){
        neighSum += s_spins[(row - 1) * blockDim.x + col];
        neighSum += s_spins[row * blockDim.x + (col - 1)];
        neighSum += s_spins[row * blockDim.x + (col + 1)];
        neighSum += s_spins[(row + 1) * blockDim.x + col];
        float dE = 2 * spin * neighSum;
        if (exp(- dE/T) > d_random[site])
            s_spins[sharedsite] = -1 * spin;
    }
    
    __syncthreads();
    neighSum = 0;
    if (chess == 1){
        neighSum += s_spins[(row - 1) * blockDim.x + col];
        neighSum += s_spins[row * blockDim.x + (col - 1)];
        neighSum += s_spins[row * blockDim.x + (col + 1)];
        neighSum += s_spins[(row + 1) * blockDim.x + col];
        float dE = 2 * spin * neighSum;
        if (exp(- dE/T) > d_random[site])
            s_spins[sharedsite] = -1 * spin;
    } 
    __syncthreads();

    //Update spins
    d_spins[site] = s_spins[sharedsite];
    __syncthreads();
    
    return;
}

__global__
void isingSample(int *d_spins, float *d_random, const float T,
                 const int L){
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bdimx = blockDim.x, bdimy = blockDim.y;
    // bdim - 2 because we don't update boundary
    int x = (int)(tidx + blockIdx.x * (bdimx - 2)) - 1;
    int y = (int)(tidy + blockIdx.y * (bdimy - 2)) - 1;
    if (x > L || y > L)
        return; //Prevents overlap with incommensurate blocks
    
    int col = MOD(x, L);
    int row = MOD(y, L);
    int site = row * L + col, sharedsite = tidy * bdimx + tidx;
    int blockChess = (blockIdx.x%2 + blockIdx.y%2)%2;
    extern __shared__ int s_spins[];//(blockDim+2)**2

    if (blockChess == 0)
        chessBoardUpdate(s_spins, d_spins, d_random, T, site, sharedsite);
    if (blockChess == 1)
        chessBoardUpdate(s_spins, d_spins, d_random, T, site, sharedsite);
    
    return;
}



