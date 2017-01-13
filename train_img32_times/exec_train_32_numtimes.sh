#!/bin/sh

# This script needs "train_img32.py" from https://github.com/Atsuto0519/Chainer-CNN_Random-seed

start_seed=0
mkdir results
for i in `seq 1 $1`
do
    echo $i times:
    python train_img32_numtimes.py -b 750 -e 102 -g 0 -o "results/result_$i" -s `expr $i + $start_seed`
    echo
done
