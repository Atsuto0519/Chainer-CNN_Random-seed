#!/bin/sh

# At first, please all datasets move train/

# Make directory
mkdir test/

# Separate train and test from all datasets
thumbnails='./train'
images_dir=(`ls $thumbnails`)
num_images_dir=${#images_dir[*]}
## Searching files in train or test
num_images_dir=`expr $num_images_dir - 1`
for i in `seq 0 $num_images_dir`
do
    echo $i
    mv $thumbnails/${images_dir[$(($i))]} $thumbnails/$i
    ls $thumbnails
    mkdir test/$i
    images_file=(`ls $thumbnails/$i`)
    num_images_file=${#images_file[*]}
    for j in `seq 0 9`
    do
	mv train/$i/${images_file[$((RANDOM%num_images_file))]} test/$i/.
    done
done
