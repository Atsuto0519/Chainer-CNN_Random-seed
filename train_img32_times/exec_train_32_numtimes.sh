#!/bin/sh

# This script needs "train_img32.py" from https://github.com/Atsuto0519/Chainer-CNN_Random-seed

start_seed=0
mkdir results
for i in `seq 1 $1`
do
    echo $i times:
    python ./groupA/train_img32_numtimes.py -b 750 -e 10 -g 0 -o "./groupA/results/result_$i" -s `expr $i + $start_seed`
    python ./groupB/train_img32_numtimes.py -b 750 -e 10 -g 0 -o "./groupB/results/result_$i" -s `expr $i + $start_seed`
    echo
done
