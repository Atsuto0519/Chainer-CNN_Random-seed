#!/bin/sh

# This script needs "train_img32.py" from https://github.com/Atsuto0519/Chainer-CNN_Random-seed

mkdir results
for i in `seq 1 $1`
do
    echo $i times:
    python ./traim_img32.py -b 100 -e 10 -g 0 -o 'result_{$i}'
done
