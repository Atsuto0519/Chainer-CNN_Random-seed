#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import numpy as np
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import training
from PIL import Image
from numpy import *
import os
import six
from chainer.dataset import download
from chainer.datasets import tuple_dataset
import matplotlib.pyplot as plt
import random

random.seed(0)

# set times
exec_times=1

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)



# Load to img of 32*32 pixel
def get_faceImage_directory(root):
    im=[]
    num_count=0
    
    for u in os.listdir(root):
        im_sub, number = get_faceImage_oneperson(root + u)
        im.extend(im_sub)
        num_count+=number

    return im, number

def get_faceImage_oneperson(root):
    num_count = 0
    im_sub = []
    im_out = []
    
    for u in os.listdir(root + '/'):
        print(u)
        # convert array from faceimage
        im = array(Image.open(root + '/' + u).convert('L'))
        im_sub.append([])
        for i in range(32):
            im_sub[num_count].extend(im[i,:])
        im_subsub=(array(im_sub[num_count]).astype(float32), root)
        im_out.append(im_subsub)
        num_count+=1
    return im_out, num_count

def get_img32(withlabel=True, ndim=1, scale=1., dtype=np.float32,
              label_dtype=np.int32):

    train_raw = _retrieve_img_training()
    train = _preprocess_img32(train_raw, withlabel, ndim, scale, dtype,
                              label_dtype)
    test_raw = _retrieve_img_test()
    test = _preprocess_img32(test_raw, withlabel, ndim, scale, dtype,
                             label_dtype)
    return train, test

def _preprocess_img32(raw, withlabel, ndim, scale, image_dtype, label_dtype):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 32, 32)
    elif ndim == 3:
        images = images.reshape(-1, 1, 32, 32)
    elif ndim != 1:
        raise ValueError('invalid ndim for IMG32 dataset')
    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images

def _retrieve_img_training():
    return _retrieve_img32('train')

def _retrieve_img_test():
    return _retrieve_img32('test')

def _retrieve_img32(name):
    root = download.get_dataset_directory('pfnet/chainer/img')
    path = os.path.join(root, name + '.npz')
    
    return download.cache_or_load_file(path, lambda path: _make_npz_img32(name), np.load)

def _make_npz_img32(name):
    im, number = get_faceImage_directory(name + '/')
    x = np.empty((number, 1024), dtype=np.uint8)
    y = np.empty(number, dtype=np.uint8)

    random.shuffle(im)
    for i in six.moves.range(number):
        x[i] = im[i][0]
        y[i] = int(im[i][1].replace(name + '/', ''))

    np.savez_compressed('/Users/inage/.chainer/dataset/pfnet/chainer/img/'+name+'.npz', x=x, y=y)
    return {'x': x, 'y': y}

def draw_image_28(data):
    dimX = 28
    dimY = 28
    Z = data.reshape(dimY,dimX)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,dimX)
    plt.ylim(0,dimY)
    plt.pcolor(Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    
def draw_image_32(data):
    dimX = 32
    dimY = 32
    Z = data.reshape(dimY,dimX)   # convert from vector to 32x32 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,dimX)
    plt.ylim(0,dimY)
    plt.pcolor(Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")


def main():
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# times: {}'.format(args.times))
    print('')
    
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(1024, args.unit, 4))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
        
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
        
    # Load the IMG32 dataset
    train, test = get_img32()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                             repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot())
    
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    
    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())
    
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    # Set exec times and numpy's random seed
    parser = argparse.ArgumentParser(description='Chainer exec: IMG32')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--times', '-t', type=int, default=1,
                        help='Number of execution times')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Number of numpy random seed')
    args = parser.parse_args()
    best_seed_val=args.seed
    
    for i in range(args.times):
        np.random.seed(i+args.seed+1)
        main()
