#!/bin/bash

gcc -O3 -fopenmp -DMATCH_OUTPUT A1_Q2_gauss_improved_cpy.c -o A1_Q2_gauss_improved_cpy

to_run=$1
num_inputs=6

echo "############ For A1_Q2 gaussian: #############"

curr=1024
for (( i=1; i<=num_inputs; i++ ))
do
    for (( j=1; j<=8; j=j*2 )) 
    do  
        avg=0
        for (( k=1; k<=to_run; k++ )) 
        do
            t=$(./A1_Q2_gauss_improved_cpy $curr  A1_Q2_outputs/A1_Q2_output$i.txt $j | cut -d":" -f 2 | cut -d" " -f 2)
            avg=$((avg + t)) 
        done
        avg=$((avg / to_run))
        echo "Average time taken on matrix dimension "$curr "by thread count "$j ":" $avg
    done 
    curr=$((curr * 2))
    #if [ $i -eq 4 ] 
    #then
    #    curr=$(( curr ))
    #fi
    echo $curr
done 
