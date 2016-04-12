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

#include <sampling.h>

//#define DBUG //Save stuff to files
#define MOD(x, N) (((x < 0) ? ((x % N) + N) : x) % N)
#ifndef MIN
#define MIN(a, b) ((a > b) ? b : a)
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? a : b)
#endif

//I actually use THREAD_X+2, THREAD_Y+2
// (THREAD_X+2)*(THREAD_Y+2) < 1024
#define THREAD_X 30 
#define THREAD_Y 30

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
			    ./cuising filename L T N_steps save_period burnin stepsize\n\
                (optional final argument) initial_spinfile";
    if (argc < 8 || argc > 9){
        printf("%s", printMSG);
	return 0;
    }

    char *filename = argv[1];
    int L = atoi(argv[2]), N_steps = atoi(argv[4]);
    float T = atof(argv[3]);
    int period = atoi(argv[5]), burnin = atoi(argv[6]);
    float step = atof(argv[7]);

    printf("Saving to %s every %d steps \nL=%d, T=%f, stepsize=%f, burnin=%d\n",
           filename, period, L, T, step, burnin);

    int N = L*L;

    curandGenerator_t rng;
    checkCudaErrors(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(rng, 920989ULL));
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t cpyStream, rngStream;
    checkCudaErrors(cudaStreamCreate(&cpyStream));
    checkCudaErrors(cudaStreamCreate(&rngStream));
    
    #ifndef DBUG
    checkCudaErrors(curandSetStream(rng, rngStream));
    #endif

    float *h_spins = (float *)malloc(sizeof(float) * N);
    float *h_mx = (float *)malloc(sizeof(float) * N);
    float *h_my = (float *)malloc(sizeof(float) * N);
    float *h_localE = (float *)malloc(sizeof(float) * N);
    float *h_ones = (float *)malloc(sizeof(float) * N);
    for (int i=0; i < N; i++)
        h_ones[i] = 1.;
    memset(h_spins, 0, sizeof(float) * N);
    memset(h_localE, 0, sizeof(float) * N);
  
    // Initialize spin configurations
    if (argc == 8){
        for (int i = 0; i < N; i++)
            h_spins[i] = 2*M_PI*(float)rand()/RAND_MAX;
    } else {
        FILE *initFile = fopen(argv[8], "r");
        if (!initFile){
            printf("no file opened, %s\n", argv[8]);
            exit(2);
        }
        int j = 0;
        for (int i = 0; i < L; i++){
            for (j = 0; j < L-1; j++)
                int err = fscanf(initFile, "%f\t", &(h_spins[i*L+j]));
            int err = fscanf(initFile, "%f\n", &(h_spins[i*L+j+1]));
        }
        fclose(initFile);
    }
    for (int i = 0; i < N; i++){
        h_mx[i] = cos(h_spins[i]); 
        h_my[i] = sin(h_spins[i]); 
    }

    float *d_spins, *d_mx, *d_my, *d_localE;
    float *d_random, *d_random_step; 
    float *d_ones;
    checkCudaErrors(cudaMalloc((void **)&d_ones, sizeof(float) * N));
    
    checkCudaErrors(cudaMalloc((void **)&d_spins, sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)&d_mx, sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)&d_my, sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)&d_localE, sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)&d_random, sizeof(float) * 2*N));

    checkCudaErrors(cudaMemcpy(d_ones, h_ones, sizeof(float) * N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spins, h_spins, sizeof(float) * N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mx, h_mx, sizeof(float) * N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_my, h_my, sizeof(float) * N, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_localE, h_localE, sizeof(float) * N, cudaMemcpyHostToDevice));

    checkCudaErrors(curandGenerateUniform(rng, d_random, 2*N));
    d_random_step = d_random + N;

    // Define CUDA blocks and memory sizes
    int BLOCKS_X = ceil((float)L/(float)THREAD_X);
    int BLOCKS_Y = ceil((float)L/(float)THREAD_Y);
    int BLOCKMEM = sizeof(float) * (THREAD_X+2) * (THREAD_Y+2);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);
    dim3 threads(THREAD_X+2, THREAD_Y+2); //include boundary

    #ifdef DBUG
    float *h_random = (float *)malloc(sizeof(float)*2*N);
    checkCudaErrors(cudaMemcpy(h_random, d_random, sizeof(float)*2*N,
                               cudaMemcpyDeviceToHost));
    FILE *fpdb = fopen("dbug.dat", "w");
    for (int i=0; i < N; i++){
        if (i%L==0)
            fprintf(fpdb, "\n");
        fprintf(fpdb, "%f\t", h_spins[i]);
    }
    for (int i=0; i < 2*N; i++){
        if (i%L==0)
            fprintf(fpdb, "\n");
        fprintf(fpdb, "%f\t", h_random[i]);
    }
    fclose(fpdb);
    printf("Blocks(%d, %d)\n", BLOCKS_X, BLOCKS_Y); 
    printf("Threads(%d, %d)\n", THREAD_X+2, THREAD_Y+2);
    #endif

    //Timing stuff
    cudaEvent_t start, stop;
    float time = 0.f;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    //Burnin
    for (int t = 0; t < burnin; t++){
        //checkCudaErrors(cudaStreamSynchronize(rngStream));
        isingSample<<<blocks, threads, 
                      BLOCKMEM>>>(d_spins, d_mx, d_my, d_localE,
                                  d_random, d_random_step, T, L, step);
        checkCudaErrors(curandGenerateUniform(rng, d_random, 2*N));
        d_random_step = d_random + N;
    } 
    printf("Burn-in complete, sampling\n");

    FILE *fpSave = fopen(filename, "w");
    FILE *fpEMag = fopen("energy_magnetization.dat", "w");

    float Mx = 0.f, My = 0.f, E = 0.f;
    for (int t = 0; t < N_steps; t++){
       
        //checkCudaErrors(cudaStreamSynchronize(rngStream));
        isingSample<<<blocks, threads, 
                      BLOCKMEM>>>(d_spins, d_mx, d_my, d_localE,
                                  d_random, d_random_step, T, L, step);
        checkCudaErrors(curandGenerateUniform(rng, d_random, 2*N));
        d_random_step = d_random + N;
        
        checkCudaErrors(cublasSdot(handle, N, d_mx, 1., d_ones, 1., &Mx));
        checkCudaErrors(cublasSdot(handle, N, d_my, 1., d_ones, 1., &My));
        checkCudaErrors(cublasSdot(handle, N, d_localE, 1., d_ones, 1., &E));
        E /= (2.*N); //double counting local energy
        Mx /= (float)N;
        My /= (float)N;
        fprintf(fpEMag, "%f\t%f\t%f\n", Mx, My, E);

        if (t % period == 0){
            //Quit changing d_spins before copying
            //checkCudaErrors(cudaStreamSynchronize(sampleStream));
            checkCudaErrors(cudaMemcpyAsync(h_spins, d_spins, sizeof(float)*N,
                                            cudaMemcpyDeviceToHost, cpyStream));
            for (int i=0; i < N; i++){
                if (i > 0 && i%L==0)
                    fprintf(fpSave, "\n");
                fprintf(fpSave, "%f\t", h_spins[i]);
            }
            fprintf(fpSave, "\n");
        }
        checkCudaErrors(cudaDeviceSynchronize());
        
    }

    fclose(fpSave);
    fclose(fpEMag);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time: %f ms, %f ms / site updated\n", time, time/(float)(burnin+N_steps+N));
    
    checkCudaErrors(cudaStreamDestroy(cpyStream));
    #ifndef DBUG
    checkCudaErrors(cudaStreamDestroy(rngStream));
    #endif
    //checkCudaErrors(cudaStreamDestroy(sampleStream));
    checkCudaErrors(curandDestroyGenerator(rng));
    cublasDestroy(handle);

    cudaFree(d_spins);
    cudaFree(d_mx);
    cudaFree(d_my);
    cudaFree(d_localE);
    cudaFree(d_random);
    cudaFree(d_ones);
    //cudaFree(d_random_step);
    free(h_spins);
    free(h_mx);
    free(h_my);
    free(h_localE);
    free(h_ones);
    checkCudaErrors(cudaGetLastError());

    return EXIT_SUCCESS;
}


