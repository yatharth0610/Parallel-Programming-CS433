#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

// C is chunk size
#define C 16

double** L;
double* y, *x;

int min(int a, int b) {
    return ((a < b) ? a : b);
}

int main(int argc, char**argv){
    FILE *in_fp, *out_fp;

    if(argc != 4){
        printf("Incorrect number of arguments! Provide input file, output file and number of threads!\n");
        exit(1);
    }
    
    // Read arguments from command line
    int n_threads = atoi(argv[3]);
    out_fp = fopen(argv[2], "w+");
    in_fp  = fopen(argv[1], "r");

    int n;
    int ret = fscanf(in_fp, "%d", &n);
    if (ret == EOF) {
        printf("Error in reading from file\n");
    }

    L = (double**)malloc(n*sizeof(double*));
    assert(L != NULL);
   
    // Read entries of L
    for(int row=0; row<n; row++){
        L[row] = (double*)malloc((row+1)*sizeof(double));
        assert(L[row] != NULL);
        for (int col = 0; col <= row; col++) {
            ret = fscanf(in_fp, "%lf", &L[row][col]);
            if (ret == EOF) {
		    exit(1);
                printf("Error in reading from file\n");
            }
        }
    }
    
    y = (double*)malloc(n*sizeof(double));
    x = (double*)malloc(n*sizeof(double));
    assert(y!=NULL);

    // Read entries of y
    for(int row=0;row<n;row++){
        ret = fscanf(in_fp, "%lf", &y[row]);
        if (ret == EOF) {
            printf("Error in reading from file\n");
        }
	x[row]=y[row];
    }

    InitializeInput();

    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    int times = (n / C) + ((n % C) ? 1 : 0);

    // Measure parallel execution time
    gettimeofday(&tv0, &tz0);

    // Iterate over chunks of columns 
#pragma omp parallel num_threads (n_threads)
    for (int i = 0; i < times; i++) {
        
        // Calculation of triangle
        #pragma omp single 
        {
            for (int col = i*C; col < min(n, (i+1)*C); col++) {
                y[col] /= L[col][col];
            
                for (int row = col + 1; row < min(n, (i+1)*C); row++) {
                    y[row] -= y[col] * (L[row][col]);
                }
            }
        }


	// Calculation of rectangle
        #pragma omp for 
        for (int row = (i+1)*C; row < n; row++) {
            for (int col = i*C; col < min(n, (i+1)*C); col++) {
                y[row] -= y[col] * L[row][col];
            }
        }

    }
    gettimeofday(&tv1, &tz1);

    // Print output to file
    for (int i = 0; i < n; i++) {
        fprintf(out_fp, "%lf ", y[i]);
    }

    printf("Time: %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec)*1000000+(tv1.tv_usec - tv0.tv_usec) );
}
