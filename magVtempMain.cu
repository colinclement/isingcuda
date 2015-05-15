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

#define THREADS_PER 30

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
			    ./cuising filename L Tmin Tmax numTs N_steps period burnin \n";
    if (argc < 9){
        printf("%s", printMSG);
	return 0;
    }
    else if (argc > 9){
        printf("%s", printMSG);
        return 0;
    }

    char *filename = argv[1];
    int L = atoi(argv[2]);
    float Tmin = atof(argv[3]), Tmax = atof(argv[4]);
    int numTs = atoi(argv[5]);
    int N_steps = atoi(argv[6]);
    int period = atoi(argv[7]), burnin = atoi(argv[8]);
    printf("Saving to %s with L=%d, Tmin=%f, Tmax=%f, numTs=%d, N_steps=%d, period=%d, burnin=%d\n", 
           filename, L, Tmin, Tmax, numTs, N_steps, period, burnin);

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

    FILE *fpMag = fopen(filename, "w");

    int NUMBLOCKS = ceil((float)L/(float)THREADS_PER);
    int BLOCKMEM = sizeof(int) * (THREADS_PER+2) * (THREADS_PER+2);
    dim3 blocks(NUMBLOCKS, NUMBLOCKS);
    dim3 threads(THREADS_PER+2, THREADS_PER+2);

    cudaEvent_t start, stop;
    float time = 0.f;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
   
    float Tstep = (Tmax - Tmin) / ((float) numTs);

    for (int iT = 0; iT < numTs; iT++){
        float temp = Tmax - Tstep * iT;
        float mag = 0.f;
        for (int t = 0; t < burnin; t++){
            isingSample<<<blocks, threads, 
                          BLOCKMEM>>>(d_spins, d_random, temp, L);
            checkCudaErrors(curandGenerateUniform(rng, d_random, N));
            checkCudaErrors(cudaDeviceSynchronize());
        } 
        
        for (int t = 0; t < N_steps; t++){
            isingSample<<<blocks, threads, 
                          BLOCKMEM>>>(d_spins, d_random, temp, L);
            checkCudaErrors(curandGenerateUniform(rng, d_random, N));
            checkCudaErrors(cudaDeviceSynchronize());
            if (t % period == 0){
                thrust::device_ptr<int> spinPtr = thrust::device_pointer_cast(d_spins);
                mag += ((float) thrust::reduce(spinPtr, spinPtr + N))/((float) N);
            }
        }
        fprintf(fpMag, "%f\t%f\n", mag/((float)N_steps/period), temp);
    }
    fclose(fpMag);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time: %f ms, %f ms/step\n", time, time/(float)burnin);
    
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

__global__
void isingSample(int *d_spins, float *d_random, const float T,
                 const int L){
    int N = L*L;
    int tidx = threadIdx.x, tidy = threadIdx.y;
    int bdimx = blockDim.x, bdimy = blockDim.y;
    int col = MOD( (int)(tidx + blockIdx.x * (bdimx - 2) - 1), L);
    int row = MOD( (int)(tidy + blockIdx.y * (bdimy - 2) - 1), L);
    int site = row * L + col, sharedsite = tidy * bdimx + tidx;
    
    if (site >= N || col >=L || row >= L)
        return;
    
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

