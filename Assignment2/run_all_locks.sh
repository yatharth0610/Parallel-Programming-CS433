#!/bin/bash

to_run=$1
flags=("PTHREAD_MUTEX" "PTHREAD_SEMAPHORE" "BAKERY" "SPIN_LOCK" "TTS" "TICKET_LOCK" "ARRAY_LOCK")

for flag in ${flags[@]}
do
    g++ -O3 -pthread -D$flag pthread_main_lock.cpp -o pthread_main_lock;
    for (( i=1; i<=8; i=i*2 ))
    do
        avg=0
        for (( j=1; j<=to_run; j++ ))
        do
            t=$(./pthread_main_lock $i | cut -d" " -f 3)
            avg=$((avg+t))
        done    
        avg=$((avg/to_run))
        echo "Average time taken for pthread with "$flag " by thread count "$i ":" $avg 
    done 
done

g++ -O3 -fopenmp omp_main_lock.cpp -o omp_main_lock;
for (( i=1; i<=8; i=i*2 ))
do
    avg=0
    for (( j=1; j<=to_run; j++ ))
    do
        t=$(./omp_main_lock $i | cut -d" " -f 3)
        avg=$((avg+t))
    done    
    avg=$((avg/to_run))
    echo "Average time taken for openmp by thread count "$i ":" $avg 
done 


