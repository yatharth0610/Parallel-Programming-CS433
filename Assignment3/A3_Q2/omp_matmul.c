#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<sys/time.h>
#include<math.h>

int min(int x, int y) {
    return (x<y? x:y);
}

int main(int argc, char*argv[]){
    int n, nthreads;
    float*A, *x, *y;
   struct timeval tv0, tv1;
   struct timezone tz0, tz1;

   if (argc != 3) {
      printf("Need matrix size (n) and number of threads (nthreads).\nAborting...\n");
      exit(1);
   }
   n = atoi(argv[1]);
   nthreads = atoi(argv[2]);
   nthreads = min(nthreads, n);
   A = (float*)malloc(sizeof(float)*(n*n));
   x = (float*)malloc(sizeof(float)*n);
   y =  (float*)malloc(sizeof(float)*n);

   #pragma omp parallel for num_threads(nthreads)
   for(int i=0; i<n; i++){
       for(int j=0; j<n; j++){
           A[n*i+j] = ((float)(i*n+j))/(n*n);
       }
   }

   #pragma omp parallel for num_threads(nthreads)
   for(int i=0; i<n; i++){
       x[i] = ((float)i)/n;
   }

   gettimeofday(&tv0, &tz0);
   #pragma omp parallel for num_threads(nthreads)
   for(int i=0; i<n; i++){
       y[i] = 0.0;
       for(int j=0; j<n; j++){
           y[i]+=A[n*i+j]*x[j];
       }
   }
   gettimeofday(&tv1, &tz1);

   printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));



}
