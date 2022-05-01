#!/bin/bash

gcc -O3 -fopenmp -DFIX A3_Q2/omp_matmul.c -o A3_Q1/omp_matmul

to_run=$1

for (( i=1024; i<=16384; i=i*2 ))
do
    for (( j=2; j<=32; j=j*2 )) 
    do  
        avg=0
        for (( k=1; k<=to_run; k++ )) 
        do
            t=$(./A3_Q2/omp_matmul $i $j | cut -d":" -f 2 | cut -d" " -f 2)
            avg=$((avg + t)) 
        done
        avg=$((avg / to_run))
        echo "Average time taken on matrix size "$i "by thread count "$j ":" $avg
    done 
done
