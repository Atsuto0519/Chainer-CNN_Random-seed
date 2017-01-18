#!/bin/sh

# This script needs "train_img32.py" from https://github.com/Atsuto0519/Chainer-CNN_Random-seed.
# Please this file in the directory has groupA and groupB.
# Directorys of groupA and groupB have train_img32.py(stable version) and train/ and test/.

mkdir results
for i in `seq 1 $1`
do
    echo $i times:
    python ./groupA/traim_img32.py -b 750 -e 10 -g 0 -o './groupA/result_$i'
    python ./groupA/traim_img32.py -b 750 -e 10 -g 0 -o './groupB/result_$i'
done
