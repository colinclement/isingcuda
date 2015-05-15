#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h> //In samples/common/inc
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//#define DBUG //Save stuff to files
#define MOD(x, N) (((x < 0) ? ((x % N) + N) : x) % N)
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif

#define THREAD_X 64
#define THREAD_Y 8

__global__
void isingSample(int *d_spins, float *d_random, const float T,
                 const int L);

__device__
void chessBoardUpdate(int *s_spins, int *d_spins, float *d_random, 
                      const float T, const int site, const int sharedsite);

int main(int argc, char **argv){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        fprintf(stderr, "Error: no CUDA supporting devices.\n");
	exit(EXIT_FAILURE);
    }
    int dev = 0; 
    cudaSetDevice(dev);
    
    const char *printMSG = "Incorrect number of arguments: Usage: \n\
			    ./cuising filename L T N_steps period burnin \n";
    if (argc < 7){
        printf("%s", printMSG);
	return 0;
    }
    else if (argc > 7){
        printf("%s", printMSG);
        return 0;
    }

    char *filename = argv[1];
    int L = atoi(argv[2]), N_steps = atoi(argv[4]);
    float T = atof(argv[3]);
    int period = atoi(argv[5]), burnin = atoi(argv[6]);
    printf("Saving to %s with L=%d, T=%f, every %d steps,\n with burnin=%d\n",
           filename, L, T, period, burnin);

    int N = L*L;

    curandGenerator_t rng;
    checkCudaErrors(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(rng, 920989ULL));

    //cublasHandle_t handle;
    //checkCudaErrors(cublasCreate(&handle));

    int *h_spins = (int *)malloc(sizeof(int) * N);
    memset(h_spins, 1, sizeof(int) * N);

    for (int i = 0; i < N; i++){
        float r = (float)rand()/RAND_MAX;
        h_spins[i] = (r > 0.5) ? 1 : -1;
    }
    int *d_spins;
    float *d_random;
    checkCudaErrors(cudaMalloc((void **)&d_spins, sizeof(int) * N));
    checkCudaErrors(cudaMalloc((void **)&d_random, sizeof(float) * N));
    checkCudaErrors(cudaMemcpy(d_spins, h_spins, sizeof(int) * N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(curandGenerateUniform(rng, d_random, N));
    checkCudaErrors(cudaDeviceSynchronize());
    float *h_random = (float *)malloc(sizeof(float) * N);
    checkCudaErrors(cudaMemcpy(h_random, d_random, sizeof(int) * N, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    FILE *fp = fopen("dbug.dat", "w");
    for (int i=0; i < N; i++){
        if (i%L ==0)
            fprintf(fp, "\n");
        fprintf(fp, "%d\t", h_spins[i]);
    }
    for (int i=0; i < N; i++){
        if (i%L ==0)
            fprintf(fp, "\n");
        fprintf(fp, "%f\t", h_random[i]);
    }

    int BLOCKS_X = ceil((float)L/(float)THREAD_X);
    int BLOCKS_Y = ceil((float)L/(float)THREAD_Y);
    int BLOCKMEM = sizeof(int) * (THREAD_X+2) * (THREAD_Y+2);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);
    dim3 threads(THREAD_X+2, THREAD_Y+2);

    cudaEvent_t start, stop;
    float time = 0.f;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
   

    for (int t = 0; t < burnin; t++){
        isingSample<<<blocks, threads, 
                      BLOCKMEM>>>(d_spins, d_random, T, L);
        checkCudaErrors(curandGenerateUniform(rng, d_random, N));
        //checkCudaErrors(cudaDeviceSynchronize());
    } 
    
    for (int t = 0; t < N_steps; t++){
        isingSample<<<blocks, threads, 
                      BLOCKMEM>>>(d_spins, d_random, T, L);
        checkCudaErrors(curandGenerateUniform(rng, d_random, N));
        //checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time: %f ms, %f ms / site updated\n", time, time/(float)(burnin+N_steps+N));
    
    checkCudaErrors(cudaMemcpy(h_spins, d_spins, sizeof(int) * N, cudaMemcpyDeviceToHost));

    for (int i=0; i < N; i++){
        if (i%L ==0)
            fprintf(fp, "\n");
        fprintf(fp, "%d\t", h_spins[i]);
    }

    fclose(fp);
    cudaFree(d_spins);
    cudaFree(d_random);
    free(h_spins);
    free(h_random);
    checkCudaErrors(cudaGetLastError());

    return EXIT_SUCCESS;
}

//TODO: CHECK BOUNDARY CONDITIONS

__global__
void isingSample(int *d_spins, float *d_random, const float T,
                 const int L){
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bdimx = blockDim.x, bdimy = blockDim.y;
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

__device__
void chessBoardUpdate(int *s_spins, int *d_spins, float *d_random, 
                      const float T, const int site, const int sharedsite){
    
    int row = threadIdx.y, col = threadIdx.x; 
    //Load spins to shared memory
    s_spins[sharedsite] = d_spins[site];
    __syncthreads();
    
    if (row == 0 || col == 0 || row == blockDim.y-1 || col == blockDim.x-1)
        return; //Edge site for shared memory filling
    
    int neighSum = 0, r = row, c = col;
    int chess = (row%2 + col%2)%2;
    int spin = s_spins[sharedsite];
    
    if (chess == 0){
        for (int i =-1; i < 2; i++){
            for (int j=-1; j < 2; j++){
                if (abs(i) != abs(j)){
                    r = row + i;
                    c = col + j;
                    neighSum += s_spins[r * blockDim.x + c];
                }
            }
        }
        float dE = 2 * spin * neighSum;
        if (exp(- dE/T) > d_random[site])
            s_spins[sharedsite] = -1 * spin;
    }
    
    __syncthreads();
    neighSum = 0;
    if (chess == 1){
        for (int i =-1; i < 2; i++){
            for (int j=-1; j < 2; j++){
                if (abs(i) != abs(j)){
                    r = row + i;
                    c = col + j;
                    neighSum += s_spins[r * blockDim.x + c];
                }
            }
        }
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

