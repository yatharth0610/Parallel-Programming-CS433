#!/bin/bash

gcc -O3 -fopenmp -DMATCH_OUTPUT A1_Q1_dp.c -o A1_Q1_dp

to_run=$1
num_inputs=$2

echo "########### For A1_Q1 dp: #############"

for (( i=9; i<=num_inputs; i++ ))
do
    for (( j=1; j<=8; j=j*2 )) 
    do  
        avg=0
        for (( k=1; k<=to_run; k++ )) 
        do
            echo $i
            t=$(./A1_Q1_dp A1_Q1_inputs/A1_Q1_input$i.txt A1_Q1_outputs/A1_Q1_output$i.txt $j | cut -d":" -f 2 | cut -d" " -f 2)
            echo $t
            avg=$((avg + t)) 
        done
        avg=$((avg / to_run))
        echo "Average time taken on test case "$i "by thread count "$j ":" $avg
    done 
done
