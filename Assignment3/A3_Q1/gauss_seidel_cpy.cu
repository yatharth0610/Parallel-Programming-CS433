#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include<sys/time.h>

#define TOL 1e-5
#define ITER_LIMIT 1000
#define NUM_THREADS_PER_BLOCK 32

__managed__ float diff = 0.0;
__managed__ int nthreads, n;
__device__ int count = 0;
__managed__ int iterations = 0;
__device__ volatile int barrier_flag = 0;

#ifdef CUDA_RANDOM
__global__ void init(unsigned int seed, curandState_t* states) {
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}

__global__ void init_kernel(float*A, int span, curandState_t* states) {    
    int id  = threadIdx.x + blockIdx.x*blockDim.x;
    int val = ((n+2)*(n+2) < span*(id+1)) ? (n+2)*(n+2) : span*(id+1);
    for (int i = span*id; i < val; i++) {
        A[i] = (curand(&states[i])%100) / 100.0;
        // A[i] = i / (n+2);
    }
}
#endif

#ifdef FIX
__global__ void init_sample_kernel(float*A, int span){
    int id  = threadIdx.x + blockIdx.x*blockDim.x;
    int val = ((n+2)*(n+2) < span*(id+1)) ? (n+2)*(n+2) : span*(id+1);
    for (int i = span*id; i < val; i++) {
        // A[i] = (curand(&states[i])%100) / 100.0;
        A[i] = float(i%100)/100;
    }

}
#endif


// call with A and nthreads
__global__ void gauss_seidel_kernel_general(float*A, int span){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int val =  n*n;
    int done = 0, row, col;
    float local_diff, temp;
    int local_sense = 0, last_count, index, iterations = 0;
    __shared__ float local_area2[NUM_THREADS_PER_BLOCK/32];

    while(!done){
        // printf("Started function!\n");
        if(id == 0){
            diff = 0.0;
        }
        local_diff = 0.0;
        
        /************ Barrier ************/
        local_sense = (local_sense ? 0 : 1);
        __syncthreads();
        last_count = atomicAdd(&count, 1);

        if (last_count == nthreads - 1) {
            count = 0;
            barrier_flag = local_sense;
        }
        while (barrier_flag != local_sense);
        
        for(int i = id; i < val; i+=span){
            row = i/(n);
            col = i-row*(n);

            index = (row+1)*(n+2)+(col+1);

            temp = A[index];
            A[index] = 0.2*(A[index+1]+A[index-1]+A[index+n+2]+A[index-n-2]+A[index]);
            local_diff += fabs(A[index]-temp);
        }

        unsigned mask = 0xffffffff;
        for (int i = warpSize/2; i > 0; i = i/2) {
            local_diff += __shfl_down_sync(mask, local_diff, i);
        }

        if(threadIdx.x % warpSize == 0) {
            local_area2[threadIdx.x/warpSize] = local_diff;
        }
        __syncthreads();

        if((threadIdx.x/(NUM_THREADS_PER_BLOCK/32)) == 0){
            local_diff = local_area2[threadIdx.x];
            for(int i=NUM_THREADS_PER_BLOCK/64; i>0; i=i/2){
                local_diff+= __shfl_down_sync(mask, local_diff, i);
            }
            if(threadIdx.x ==0){
                atomicAdd(&diff, local_diff);
            }

        }
        /************ Barrier ************/
        local_sense = (local_sense ? 0 : 1);
        __syncthreads();
        last_count = atomicAdd(&count, 1);
        if (last_count == nthreads - 1) {
            count = 0;
            barrier_flag = local_sense;
        }
        while (barrier_flag != local_sense);

        iterations++;
        if ((diff/(n*n)< TOL) || (iterations == ITER_LIMIT)){
            done = 1;
        }

        /************ Barrier ************/
        local_sense = (local_sense ? 0 : 1);
        __syncthreads();
        last_count = atomicAdd(&count, 1);
        if (last_count == nthreads - 1) {
            count = 0;
            barrier_flag = local_sense;
        }
        while (barrier_flag != local_sense);
    }
    if(id == 0){
        printf("Iterations : %d\n", iterations);
    }
}


__global__ void gauss_seidel_kernel(float*A, int span){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int val = ((n+2)*(n+2) < span*(id+1)) ? (n+2)*(n+2) : span*(id+1);
    int done = 0, row, col;
    float local_diff, temp;
    int local_sense = 0, last_count;
    __shared__ float local_area2[NUM_THREADS_PER_BLOCK/32];


    while(!done){
        // printf("Started function!\n");
        if(id == 0){
            diff = 0.0;
        }
        local_diff = 0.0;
        
        /************ Barrier ************/
        local_sense = (local_sense ? 0 : 1);
        __syncthreads();
        last_count = atomicAdd(&count, 1);

        if (last_count == nthreads - 1) {
            count = 0;
            barrier_flag = local_sense;
        }
        while (barrier_flag != local_sense);
        
        for(int i = span*id; i < val; i++){
            row = i/(n+2);
            col = i-row*(n+2);

            if (row != 0 && row != n+1 && col != 0 and col != n+1) {
                temp = A[i];
                A[i] = 0.2*(A[i] + A[i-1] + A[i+1] + A[i + n + 2] + A[i - n - 2]);
                local_diff += fabs(A[i] - temp); 
            }
        }

        unsigned mask = 0xffffffff;
        for (int i = warpSize/2; i > 0; i = i/2) {
            local_diff += __shfl_down_sync(mask, local_diff, i);
        }

        if(threadIdx.x % warpSize == 0) {
            local_area2[threadIdx.x/warpSize] = local_diff;
        }
        __syncthreads();

        if((threadIdx.x/(NUM_THREADS_PER_BLOCK/32)) == 0){
            local_diff = local_area2[threadIdx.x];
            for(int i=NUM_THREADS_PER_BLOCK/64; i>0; i=i/2){
                local_diff+= __shfl_down_sync(mask, local_diff, i);
            }
            if(threadIdx.x ==0){
                atomicAdd(&diff, local_diff);
            }

        }

        // if (id == 0) {
        //     atomicAdd(&diff, local_diff);
        // }
        
        /************ Barrier ************/
        local_sense = (local_sense ? 0 : 1);
        __syncthreads();
        last_count = atomicAdd(&count, 1);
        if (last_count == nthreads - 1) {
            count = 0;
            barrier_flag = local_sense;
        }
        while (barrier_flag != local_sense);

        iterations++;
        if ((diff/(n*n)< TOL) || (iterations == ITER_LIMIT)){
            done = 1;
        }

        /************ Barrier ************/
        local_sense = (local_sense ? 0 : 1);
        __syncthreads();
        last_count = atomicAdd(&count, 1);
        if (last_count == nthreads - 1) {
            count = 0;
            barrier_flag = local_sense;
        }
        while (barrier_flag != local_sense);
    }
}

int main(int argc, char*argv[]){
    float*A;
    struct timeval tv0, tv2;
    struct timezone tz0, tz2;
    cudaError_t err;

    if(argc!=3){
        printf("Need dimensions of grid and number of threads\n");
        exit(1);
    }
    n = atoi(argv[1]);
    nthreads = atoi(argv[2]);

    cudaMallocManaged((void**)&A, sizeof(float)*(n+2)*(n+2));

    int device = -1;
    cudaGetDevice(&device);
	cudaMemAdvise(A, sizeof(float)*(n+2)*(n+2), cudaMemAdviseSetPreferredLocation, device);

    unsigned long span = (n+2)*(n+2)/nthreads;
    if (span*nthreads < (n+2)*(n+2)){
        span++;
    }

    #ifdef FIX
    init_sample_kernel<<< nthreads/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK >>>(A,span);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      		printf("CUDA Error2: %s\n", cudaGetErrorString(err));
      		exit(-1);
   	}
    #endif

    #ifdef CUDA_RANDOM

    curandState_t* states;

    cudaMalloc((void**) &states, (n+2) * (n+2) * sizeof(curandState_t));

    init<<<(n+2)*(n+2), 1>>>(time(0), states);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      		printf("CUDA ErrorINIT: %s\n", cudaGetErrorString(err));
      		exit(-1);
   	}
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      		printf("CUDA ErrorSYNC: %s\n", cudaGetErrorString(err));
      		exit(-1);
    }

    
    init_kernel<<< nthreads/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK >>>(A, span, states);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      		printf("CUDA Error1: %s\n", cudaGetErrorString(err));
      		exit(-1);
   	}
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      		printf("CUDA Error2: %s\n", cudaGetErrorString(err));
      		exit(-1);

   	}
    #endif
    printf("Matrix initalization done!\n");

    gettimeofday(&tv0, &tz0);
    printf("Calling kernel!\n");
    gauss_seidel_kernel<<< nthreads/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK >>>(A, span);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      		printf("CUDA Error3: %s\n", cudaGetErrorString(err));
      		exit(-1);
   	}
    cudaDeviceSynchronize();
    err = cudaGetLastError();        // Get error code

    if ( err != cudaSuccess ) {
            printf("CUDA Error4: %s\n", cudaGetErrorString(err));
            exit(-1);
    }

    gettimeofday(&tv2, &tz2);
    printf("Time: %ld microseconds diff : %f iterations: %d\n", (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec), diff / (n*n), iterations);
}
