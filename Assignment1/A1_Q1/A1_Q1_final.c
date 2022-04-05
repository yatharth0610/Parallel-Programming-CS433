#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

int** graph;
int N, min_cost, last_index;
long long int M;
int* min_cost_path, *done;
int **dp;
int** subsets, **parent;

int min(int a, int b) {
    return (a < b) ? a : b;
}

void Solve(){

    //For subset size = 2 (base case)
    for(int i=1; i<N ; i++){
        dp[(1LL<<i) + 1][i] = graph[0][i];
        parent[(1LL<<i) + 1][i] = 0;
    }

    #pragma omp parallel
    for(int subset_size = 3; subset_size <= N; subset_size++){

        #pragma omp for 
        for(long long int subset_index = 0; subset_index < done[subset_size]; subset_index++){
            long long int subset = subsets[subset_size][subset_index];

	    //iterate over last visited node in subset
            for(int last = 1; last < N; last++){
                dp[subset][last] = 1e9;
                if(!(subset&(1LL<<last))) continue;

		//subset representation before visiting last node
                long long int prev = subset - (1LL<<last);

		//iterate over second last visited node in subset
                for(int second_last=1; second_last < N; second_last++){
                    if(!(prev&(1LL<<second_last))) continue;
                    if( dp[subset][last] > dp[prev][second_last]+graph[second_last][last]){
                        dp[subset][last] = dp[prev][second_last]+graph[second_last][last];
                        parent[subset][last] = second_last;
                    }
                }
            }
        }
    }

    //find the last visited in min cost path
    for(int i=1;i<N;i++){
        if(min_cost > dp[M-1][i]+graph[i][0]){
            min_cost = dp[M-1][i]+graph[i][0];
            last_index = i;
        }
    }

}

// Function that generates the min-cost path
void GenPath(){
    min_cost_path[N-1] = last_index;
    long long curr_subset = M-1;
    int last = last_index;
    for(int i=N-2;i>=0; i--){
        int second_last = parent[curr_subset][last];
        min_cost_path[i] = second_last;
        curr_subset -= 1LL<<last;
        last = second_last;
    }
}

int main(int argc, char**argv) {
    FILE *in_fp, *out_fp;

    if(argc != 4){
        printf("Incorrect number of arguments! Provide input file, output file and number of threads!\n");
        exit(1);
    }
    
    // Read arguments from command line
    int n_threads = atoi(argv[3]);
    in_fp = fopen(argv[1], "r");
    out_fp = fopen(argv[2], "w+");

    int ret = fscanf(in_fp, "%d", &N);
    if (ret == EOF) {
        printf("Error in reading from file\n");
    }

    graph = (int**)malloc(N*sizeof(int*));

    // Read edge weights from input file
    for (int i = 0; i < N; i++) {
        graph[i] = (int*)malloc(N*sizeof(int));
        graph[i][i] = 0;
        for (int j = i+1; j < N; j++) {
            ret = fscanf(in_fp, "%d", &graph[i][j]);
            if (ret == EOF) {
                printf("Error in reading from file\n");
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            graph[i][j] = graph[j][i];
        }
    }

    min_cost_path = (int*)malloc(N * sizeof(int));
    min_cost = 1e9;
    M = ((1LL)<<N);

    // Precompute the sizes of subsets
    int ncr[N+1][N+1];
    for (int i = 0; i <= N; i++) {
         ncr[i][0] = 1;
	 ncr[i][i] = 1;
    }

    for(int i = 2; i <= N ; i++){
	    for(int j=1 ; j<i ; j++){
		    ncr[i][j] = ncr[i-1][j-1] + ncr[i-1][j];
	    }
    }

    // Array to store the minimum cost for a set of visited vertices with a given last visited vertex
    dp = (int**)malloc(M*sizeof(int*));
    parent= (int**)malloc(M*sizeof(int*));
    subsets = (int**)malloc((N+1)*sizeof(int*));
    done = (int*)malloc((N+1) * sizeof(int));
    for (int i = 0; i <= N; i++) {
        subsets[i] = (int*)malloc(ncr[N][i] * sizeof(int));
        done[i] = 0;
    }

    // Storing subsets with a given number of 1s
    for (long long int i = 3; i < M; i+=2) {
        int c_ones = 0;
        for (int j = 0; j < 64; j++) {
            if (((1LL << j) & i)) {
                c_ones++;
            }
        }
        subsets[c_ones][done[c_ones]] = i;
        done[c_ones]++;
    }
    
    for(long long int i=0; i<M; i++){
        dp[i] = (int*)malloc(N*sizeof(int));
        parent[i] = (int*)malloc(N*sizeof(int));
    }

    for(long long int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            dp[i][j] = -1;
        }
    }

    // Setting the number of threads to be used in the program
    omp_set_num_threads(n_threads);

    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    
    // Measuring execution time of parallel code
    gettimeofday(&tv0, &tz0);
    Solve();
    gettimeofday(&tv1, &tz1);
    
    GenPath();

    // Writing output to the output file
    for (int i = 0; i < N; i++) {
        fprintf(out_fp, "%d ", min_cost_path[i]);
    }
    fprintf(out_fp, "\n");
    fprintf(out_fp, "%d", min_cost);

    printf("Time: %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec)*1000000+(tv1.tv_usec - tv0.tv_usec) );

}
