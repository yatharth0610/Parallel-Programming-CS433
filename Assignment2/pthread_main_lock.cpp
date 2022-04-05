#include "sync_library.cpp"
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 10000000
#define BLOCK_SIZE 64

int x = 0, y = 0;
int num_threads;

#ifdef PTHREAD_MUTEX
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Acquire_pthread_lock(&mutex);
        assert(x == y);
        x = y+1;
        y++;
        Release_pthread_lock(&mutex);
    }
    return NULL;
}
#endif

#ifdef PTHREAD_SEMAPHORE
sem_t semaphore;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Acquire_semaphore_lock(&semaphore);
        assert(x == y);
        x = y+1;
        y++;
        Release_semaphore_lock(&semaphore);
    }
    return NULL;
}
#endif

#ifdef BAKERY
volatile int* choosing, *ticket;
int step;

void *solve(void* param) {
    int id = *(int*)param;

    for(int i=0; i<N; i++){
	Acquire_bakery_lock(choosing, ticket, id, num_threads, step);
        assert(x == y);
        x = y+1;
        y++;
        Release_bakery_lock(ticket, id, step);
    }
    return NULL;
}
#endif

#ifdef SPIN_LOCK
int lock = 0;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Acquire_spin_lock(&lock);
        assert(x == y);
        x = y+1;
        y++;
        Release_spin_lock(&lock);
    }
    return NULL;
}
#endif

#ifdef TTS
volatile int lock = 0;

void *solve(void* param) {
    int id = *(int*)param;
    for(int i=0; i<N; i++){
        Acquire_tts_lock(&lock);
        assert(x == y);
        x = y+1;
        y++;
        Release_tts_lock(&lock);
    }
    return NULL;
}
#endif

#ifdef TICKET_LOCK
volatile int ticket = 0;
volatile int release_count = 0;

void *solve(void*) {
    for(int i=0; i<N; i++){
        Acquire_ticket_lock(&ticket, &release_count);
        assert(x == y);
        x = y+1;
        y++;
        Release_ticket_lock(&release_count);
    }
    return NULL;
}
#endif

#ifdef ARRAY_LOCK
volatile int *arr;
volatile int ticket = 0;
int step;

void *solve(void*) {
    for(int i=0; i<N; i++){
        int my_id = Acquire_array_lock(arr, &ticket, step, num_threads);
        assert(x == y);
        x = y+1;
        y++;
        Release_array_lock(arr, my_id, step, num_threads);
    }
    return NULL;
}
#endif

int main(int argc, char **argv) {

    pthread_t *tid;
    pthread_attr_t attr;
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    if (argc!=2){
        printf("Need number of threads\n");
    }

    num_threads = atoi(argv[1]);

    #ifdef PTHREAD_MUTEX
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    pthread_mutex_destroy(&mutex);
    #endif 

    #ifdef PTHREAD_SEMAPHORE
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    sem_init(&semaphore, 0, 1);

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    sem_destroy(&semaphore);
    #endif

    #ifdef BAKERY
    int *id;
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    id = (int*)malloc(num_threads*sizeof(int));
    pthread_attr_init(&attr);
    
    step = BLOCK_SIZE/sizeof(int);
    choosing = (int*)malloc(num_threads*sizeof(int)*step);
    ticket = (int*)malloc(num_threads*sizeof(int)*step);

    for (int i = 0; i < num_threads; i++){
        choosing[i*step] = 0;
        ticket[i*step] = 0;
    }

    for (int i = 0; i < num_threads; i++) {
        id[i] = i;
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

    #ifdef SPIN_LOCK
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif 

    #ifdef TTS
    int *id;
    id = (int*)malloc(num_threads*sizeof(int));
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    for (int i = 0; i < num_threads; i++) {
        id[i] = i;
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

    #ifdef TICKET_LOCK
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif 

    #ifdef ARRAY_LOCK
    tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    pthread_attr_init(&attr);
    
    step = BLOCK_SIZE/sizeof(int);
    arr = (int*)malloc(num_threads*sizeof(int)*step);

    for (int i = 0; i < num_threads; i++){
        arr[i*step] = 0;
    }

    arr[0] = 1;

    gettimeofday(&tv0, &tz0);
    for(int i=0; i<num_threads; i++){
        pthread_create(&tid[i], &attr, solve, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    gettimeofday(&tv1, &tz1);
    #endif

    assert(x == y);
    assert(x == N*num_threads);
    
    printf("It took %ld microseconds for %d threads\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), num_threads);
}
