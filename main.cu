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
			    ./cuising filename L T N_steps period burnin stepsize\n";
    if (argc != 8){
        printf("%s", printMSG);
	return 0;
    }
    //else if (argc > 8){
    //    printf("%s", printMSG);
    //    return 0;
    //}

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

    cudaStream_t cpyStream, rngStream;
    checkCudaErrors(cudaStreamCreate(&cpyStream));
    checkCudaErrors(cudaStreamCreate(&rngStream));
    
    #ifndef DBUG
    checkCudaErrors(curandSetStream(rng, rngStream));
    #endif

    float *h_spins = (float *)malloc(sizeof(float) * N);
    memset(h_spins, 0, sizeof(float) * N);
    
    //Start with random angles
    for (int i = 0; i < N; i++){
        h_spins[i] = 2*M_PI*(float)rand()/RAND_MAX;
    }

    float *d_spins;
    float *d_random, *d_random_step; 
    checkCudaErrors(cudaMalloc((void **)&d_spins, sizeof(float) * N));
    checkCudaErrors(cudaMalloc((void **)&d_random, sizeof(float) * 2*N));
    checkCudaErrors(cudaMemcpy(d_spins, h_spins, sizeof(float) * N, cudaMemcpyHostToDevice));

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
                      BLOCKMEM>>>(d_spins, d_random, d_random_step, T, L, step);
        checkCudaErrors(curandGenerateUniform(rng, d_random, 2*N));
        d_random_step = d_random + N;
    } 
    printf("Burn-in complete, sampling\n");

    FILE *fpSave = fopen(filename, "w");

    for (int t = 0; t < N_steps; t++){
       
        //checkCudaErrors(cudaStreamSynchronize(rngStream));
        isingSample<<<blocks, threads, 
                      BLOCKMEM>>>(d_spins, d_random, d_random_step, T, L, step);
        checkCudaErrors(curandGenerateUniform(rng, d_random, 2*N));
        d_random_step = d_random + N;
        
        if (t % period == 0){
            //Quit changing d_spins before copying
            //checkCudaErrors(cudaDeviceSynchronize());
            //checkCudaErrors(cudaStreamSynchronize(sampleStream));
            checkCudaErrors(cudaMemcpyAsync(h_spins, d_spins, sizeof(float)*N,
                                            cudaMemcpyDeviceToHost, cpyStream));
            //checkCudaErrors(cudaMemcpy(h_spins, d_spins, sizeof(float)*N,
            //                           cudaMemcpyDeviceToHost));
            for (int i=0; i < N; i++){
                if (i%L==0)
                    fprintf(fpSave, "\n");
                fprintf(fpSave, "%f\t", h_spins[i]);
            }
            fprintf(fpSave, "\n");
        }
        //checkCudaErrors(cudaDeviceSynchronize());
        
    }

    fclose(fpSave);

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

    cudaFree(d_spins);
    cudaFree(d_random);
    //cudaFree(d_random_step);
    free(h_spins);
    checkCudaErrors(cudaGetLastError());

    return EXIT_SUCCESS;
}


