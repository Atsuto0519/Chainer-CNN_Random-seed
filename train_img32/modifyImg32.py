import chainer
from chainer import training
from PIL import Image
from numpy import *
import numpy as np
import os
import six
from chainer.dataset import download
from chainer.datasets import tuple_dataset
import matplotlib.pyplot as plt

random.seed(0)
train, test = chainer.datasets.get_mnist()

print(len(train))
print(len(train[0]))
print("train[0][0] =", train[0][0])
print("train[0][1] =", train[0][1])
print()

for i in range(10):
    print("train[%d][0] = %d" % (i, int(train[i][1])))
print()

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
    random.shuffle(im)
    x = np.empty((number, 1024), dtype=np.uint8)
    y = np.empty(number, dtype=np.uint8)

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




#index = 7
#print("label=%d" % (train[index][1]))
#draw_image_28(train[index][0])
#plt.show()

name = 'train/'
im , number = get_faceImage_directory(name)
draw_image_32(im[0][0])
plt.show()
