#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <assert.h>

using namespace std;

/*********************** GLOBAL STRUCTS *****************************/
struct bar_name{
    int counter;
    volatile int flag;
    pthread_mutex_t lock;
};

struct bar_name_conditional{
    int counter;
    pthread_cond_t cv;
    pthread_mutex_t lock;
};

/************************* BAKERY LOCK ****************************/

/* Acquire function for bakery lock */
void Acquire_bakery_lock(volatile int* choosing, volatile int*ticket, int id, int num_threads, int step){
    
    choosing[id*step] = 1;
    asm("mfence":::"memory");

    int maxval = 0, maxi = num_threads*step, id_step = id*step;

    for(int i=0; i < maxi; i+=step){
        if (maxval < ticket[i]) {
            maxval = ticket[i];
        }
    }
    ticket[id_step] = maxval+1;
    asm("mfence":::"memory");

    choosing[id_step] = 0;

    for(int j=0; j < maxi; j+=step) {
        while(choosing[j]);
        while(ticket[j] && (ticket[j] < ticket[id_step] || ((ticket[j]==ticket[id_step]) && (j<id_step))));
    }
}

/* Release function for bakery lock */
void Release_bakery_lock(volatile int*ticket, int id, int step){
   asm("":::"memory");
   ticket[id*step] = 0;
}


/************************* SPIN LOCK ****************************/

unsigned char CompareAndSet(int oldVal, int newVal, int*ptr){
    int oldValOut;
    unsigned char result;
    asm("lock cmpxchgl %4, %1 \n setzb %0"
         : "=qm"(result), "+m"(*ptr), "=a"(oldValOut) 
         : "a"(oldVal), "r"(newVal)
         : );
    return result;
}

/* Acquire function for spin lock */
void Acquire_spin_lock(int* lock){
    while(!CompareAndSet(0, 1, lock));
}

/* Release function for spin lock */
void Release_spin_lock(int *lock){
    asm("":::"memory");
    *lock = 0;
}


/************************* TEST AND TEST AND SET LOCK ****************************/

unsigned char TestAndSet(volatile int *ptr){
    int val = 1;
    asm("lock xchg %0, %1" : "+q"(val), "+m"(*ptr));
    return !val;
}

/* Acquire function for test and test and set lock */
void Acquire_tts_lock(volatile int* lock){
    while (!TestAndSet(lock)) {
        while (*lock != 0);
    }
}

/* Release function for test and test and set lock */
void Release_tts_lock(volatile int *lock){
    // assert(*lock == 1);
    asm("":::"memory");
    *lock = 0;
}

/************************* TICKET LOCK ****************************/

int FetchIncCompSet(int oldVal, int newVal, volatile int*ptr){
    int oldValOut;
    unsigned char result;
    asm("lock cmpxchgl %4, %1 \n setzb %0"
         : "=qm"(result), "+m"(*ptr), "=a"(oldValOut) 
         : "a"(oldVal), "r"(newVal)
         : );
    return  oldValOut;
}

int FetchAndInc(volatile int*x){
    int y;
    do{
        y = *x;
    }
    while(y != FetchIncCompSet(y, y+1, x));
    return y;
}

/* Acquire function for ticket lock */
void Acquire_ticket_lock(volatile int *ticket, volatile int *release_count){
    int x = FetchAndInc(ticket);
    while(*release_count != x);
}

/* Release function for ticket lock */
void Release_ticket_lock(volatile int *release_count){
    asm("":::"memory");
    *release_count = *release_count + 1;
}

/************************* ARRAY LOCK ****************************/

/* Acquire function for array lock */
int Acquire_array_lock(volatile int*arr, volatile int*ticket, int step, int num_threads){
    int x = FetchAndInc(ticket)%num_threads;
    int index = x*step;
    while(arr[index]!= 1);
    return x;
}

/* Release function for array lock */
void Release_array_lock(volatile int*arr, int id, int step, int num_threads){
    asm("":::"memory");
    int next_id = (id+1)%num_threads;
    arr[id*step] = 0;
    arr[next_id*step] = 1;
}

/************************* POSIX LOCK ****************************/

/* Acquire function for POSIX mutex */
void Acquire_pthread_lock(pthread_mutex_t* mutex){
    pthread_mutex_lock(mutex);

}

/* Release function for POSIX mutex */
void Release_pthread_lock(pthread_mutex_t* mutex){
    pthread_mutex_unlock(mutex);
}

/************************* BINARY SEMAPHORE LOCK ****************************/

/* Acquire function for Semaphore lock */
void Acquire_semaphore_lock(sem_t* semaphore){
    sem_wait(semaphore);
}

/* Release function for Semaphore lock */
void Release_semaphore_lock(sem_t* semaphore){
    sem_post(semaphore);
}


/********************* SENSE REVERSING USING BUSY WAIT *************************/

void Barrier_sense_reversal(struct bar_name* bar, int num_threads){
    int local_sense = !bar->flag;
    pthread_mutex_lock(&(bar->lock));
    bar->counter++;
    if(bar->counter == num_threads){
        pthread_mutex_unlock(&(bar->lock));
        bar->counter = 0;
        asm("mfence":::"memory");
        bar->flag = local_sense;
    }
    else {
        pthread_mutex_unlock(&(bar->lock));
        while(bar->flag != local_sense);
    }
}

/********************* TREE BARRIER USING BUSY WAIT *************************/

void Barrier_tree_busy_wait(int id, int num_threads, int height, volatile int **flags){
    unsigned int i, mask;
    for (i = 0, mask = 1; (mask & id) != 0; i++, mask <<= 1) {
        while (!flags[id][i]);
        flags[id][i] = 0;
    }
    
    if (id < (num_threads - 1)){
        flags[id+mask][i] = 1;
        while(!flags[id][height]);
        flags[id][height] = 0;
    }

    for (mask >>= 1; mask > 0; mask >>= 1) {
        flags[id - mask][height] = 1;
    }
}

/********************* POSIX CONDITIONAL *************************/

void Barrier_centralised_conditional(struct bar_name_conditional*bar, int num_threads){
    pthread_mutex_lock(&(bar->lock));
    bar->counter++;
    if(bar->counter == num_threads){
        bar->counter = 0;
        pthread_cond_broadcast(&(bar->cv));
    }
    else pthread_cond_wait(&(bar->cv), &(bar->lock));
    pthread_mutex_unlock(&(bar->lock));
}

/*********************  TREE BARRIER USING POSIX CONDITIONAL *************************/

void Barrier_tree_conditional(int id, int num_threads, int height, pthread_cond_t**cv, pthread_mutex_t *lock, volatile int** flags){
    unsigned int i, mask;

    for (i = 0, mask = 1; (mask & id) != 0; i++, mask <<= 1) {
        pthread_mutex_lock(&lock[id]);
        while (!flags[id][i]){
            pthread_cond_wait(&cv[id][i], &lock[id]);
        }
        pthread_mutex_unlock(&lock[id]);
        flags[id][i] = 0;        
    }

    if (id < (num_threads - 1)){
        pthread_mutex_lock(&lock[id+mask]);
        flags[id+mask][i] = 1;
        pthread_mutex_unlock(&lock[id+mask]);
        pthread_cond_signal(&cv[id+mask][i]);
        pthread_mutex_lock(&lock[id]);
        while(!flags[id][height]){
            pthread_cond_wait(&cv[id][height], &lock[id]);
        }
        pthread_mutex_unlock(&lock[id]);
        flags[id][height] = 0;
    }

    for (mask >>= 1; mask > 0; mask >>= 1) {
        pthread_mutex_lock(&lock[id-mask]);
        flags[id-mask][height] = 1;
        pthread_mutex_unlock(&lock[id-mask]);
        pthread_cond_signal(&cv[id-mask][height]);
    }

}

/********************* PTHREAD BARRIER  *************************/

void Barrier_pthread(pthread_barrier_t *barrier){
    pthread_barrier_wait(barrier);
}


