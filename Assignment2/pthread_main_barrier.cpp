#include "sync_library.cpp"
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <math.h>

#define N 1000000
#define BLOCK_SIZE 64

int num_threads;

#ifdef PTHREAD_BARRIER
pthread_barrier_t barrier;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Barrier_pthread(&barrier);
    }
    return NULL;
}

#endif

#ifdef SENSE_REVERSAL
struct bar_name bar;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Barrier_sense_reversal(&bar, num_threads);
    }
    return NULL;
}
#endif 

#ifdef TREE_BUSY_WAIT

volatile int**flags;
int height;
void *solve(void* params) {
    int id = *((int*)params);
    for(int i=0; i<N; i++){
        Barrier_tree_busy_wait(id, num_threads, height, flags);
    }
    return NULL;
}
#endif 

#ifdef CENTRALISED_CONDITIONAL
struct bar_name_conditional bar;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Barrier_centralised_conditional(&bar, num_threads);
    }
    return NULL;
}
#endif

#ifdef TREE_CONDITIONAL

pthread_cond_t**cv;
pthread_mutex_t* lock;
volatile int **flags;
int height;

void *solve(void* params) {
    int id = *((int*)params);
    for(int i=0; i<N; i++){
	Barrier_tree_conditional(id, num_threads, height, cv, lock, flags);
    }
    return NULL;
}
#endif

int main(int argc, char **argv){
    pthread_t *tid;
    pthread_attr_t attr;
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    if (argc!=2){
        printf("Need number of threads\n");
    }

    num_threads = atoi(argv[1]);

    #ifdef PTHREAD_BARRIER
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    pthread_barrier_init (&barrier, NULL, num_threads);

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif 

    #ifdef SENSE_REVERSAL
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    bar.counter = 0;
    bar.flag = 0;
    bar.lock = PTHREAD_MUTEX_INITIALIZER;

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif 

    #ifdef TREE_BUSY_WAIT
    int* id = (int*)malloc(num_threads*sizeof(int));
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    for (int i = 0; i < num_threads; i++) {
        id[i] = i;
    }

    flags = (volatile int**)malloc(num_threads*sizeof(int*));
    height = log2(num_threads)-1;

    for(int i = 0; i < num_threads; i++){
        flags[i] = (int*)malloc((height+1)*sizeof(int));
    }

    for(int i=0;i<num_threads;i++){
        for(int j=0;j<=height; j++){
            flags[i][j] = 0;
        }
    }

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, &id[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif

    #ifdef CENTRALISED_CONDITIONAL
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    bar.counter = 0;
    pthread_cond_init(&(bar.cv), NULL);
    bar.lock = PTHREAD_MUTEX_INITIALIZER;

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif 
    
    #ifdef TREE_CONDITIONAL
    int* id = (int*)malloc(num_threads*sizeof(int));
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    for (int i = 0; i < num_threads; i++) {
        id[i] = i;
    }

    cv = (pthread_cond_t**)malloc(num_threads*sizeof(pthread_cond_t*));
    lock = (pthread_mutex_t*)malloc(num_threads*sizeof(pthread_mutex_t));
    height = log2(num_threads)-1;

    flags = (volatile int**)malloc(num_threads*sizeof(int*));

    for(int i = 0; i < num_threads; i++){
        flags[i] = (int*)malloc((height+1)*sizeof(int));
    }

   for(int i = 0 ;i < num_threads; i++){
        for(int j = 0; j <= height; j++){
            flags[i][j] = 0;
        }
    }

    for(int i = 0; i < num_threads; i++){
        cv[i] = (pthread_cond_t*)malloc((height+1)*sizeof(pthread_cond_t));
    }

    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j <= height; j++) {
            pthread_cond_init(&cv[i][j], NULL);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        lock[i] = PTHREAD_MUTEX_INITIALIZER;
    }

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, &id[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif

    printf("It took %ld microseconds for %d threads\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), num_threads);
}

    
