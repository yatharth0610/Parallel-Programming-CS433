#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#include<assert.h>

#define THREADS_PER_BLOCK 16
#define TILE_SIZE 16
#define TOL 1e-5

__managed__ unsigned long n, nthreads;

__global__ void init_kernel(float*A, int span, int size){
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    /* add min in case not multiple */
    for(int i= id*span; i<(id+1)*span; i++){
        A[i] = (float)i/size;
    }
}

__global__ void mult_kernel(float*A, float*x, float*y, int span){

   int id = threadIdx.x + blockIdx.x*blockDim.x;
   int row;
   printf("id: %d, span: %d\n", id, span);
   for(row = id; row < n; row+=span){
       //printf("%d\n", row);
       y[row] = 0.0;
       for(int tile_num=0; tile_num<n; tile_num++){
            y[row]+=A[n*row+tile_num]*x[tile_num];
           
       }
       //printf("id: %d,  row: %f, index: %d\n", id, y[row], row);
       //__syncthreads();
   }

}

float calc_diff(float*A, float*x, float*y){
    float*y_cpu = (float*)malloc(sizeof(float)*n);
    for(int i=0; i<n; i++){
        y_cpu[i] = 0.0;
        for(int j=0; j<n; j++){
            y_cpu[i]+= A[n*i+j]*x[j];
        }
    }
    float err = 0.0;
    for(int i=0; i<n; i++){
        err+=fabs(y_cpu[i]-y[i]);
        /*if(fabs(y_cpu[i]-y[i]) > TOL){
           printf("%f %f %f\n", y_cpu[i], y[i], fabs(y[i]-y_cpu[i]));
        }*/
    }
    return err/n;
}


int main(int argc, char*argv[]){
    float*A, *x, *y;
    struct timeval tv0, tv2;
    struct timezone tz0, tz2;
    cudaError_t err;

    if(argc!=3){
        printf("Need dimensions and number of threads\n");
        exit(1);
    }

    n =atoi(argv[1]);
    nthreads =atoi(argv[2]);

    cudaMallocManaged((void**)&A, sizeof(float)*n*n);
    cudaMallocManaged((void**)&x, sizeof(float)*n);
    cudaMallocManaged((void**)&y, sizeof(float)*n);
    
    // Preferred location not set

    init_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(A,n*n/nthreads,n*n);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      	printf("CUDA ErrorFirstInit: %s\n", cudaGetErrorString(err));
      	exit(-1);
   	}
    init_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, n/nthreads, n);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      	printf("CUDA ErrorSecondInit: %s\n", cudaGetErrorString(err));
      	exit(-1);
   	}

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      	printf("CUDA ErrorSynchronise1: %s\n", cudaGetErrorString(err));
      	exit(-1);
   	}

    gettimeofday(&tv0, &tz0);

    mult_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(A, x, y, nthreads);
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      	printf("CUDA ErrorFMultKernel: %s\n", cudaGetErrorString(err));
      	exit(-1);
   	}

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if ( err != cudaSuccess ) {
      	printf("CUDA ErrorSynchronize2: %s\n", cudaGetErrorString(err));
      	exit(-1);
   	}

    gettimeofday(&tv2, &tz2);
    float avg_err = calc_diff(A, x, y);
    printf("Time: %ld microseconds and average error: %f\n", (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec), avg_err);





}
