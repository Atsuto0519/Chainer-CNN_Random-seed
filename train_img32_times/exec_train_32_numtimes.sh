#!/bin/sh

# This script needs "train_img32.py" from https://github.com/Atsuto0519/Chainer-CNN_Random-seed.
# Please this file in the directory has groupA and groupB.
# Directorys of groupA and groupB have train_img32.py(stable version) and train/ and test/.

cd groupA/
mkdir results
rm $HOME/.chainer/dataset/pfnet/chainer/img/*
for i in `seq 1 $1`
do
    echo $i times:
    python train_img32.py -b 750 -e 10 -o results/result_$i -s $i
    echo
done

cd ../groupB/
mkdir results
rm $HOME/.chainer/dataset/pfnet/chainer/img/*
for i in `seq 1 $1`
do
    echo $i times:
    python train_img32.py -b 750 -e 10 -o results/result_$i -s $i
    echo
done
