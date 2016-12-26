import chainer
from chainer import training
from chainer import function
import matplotlib.pyplot as plt

def draw_image(data, cnt):
    dimX = 28
    dimY = 28
    plt.subplot(10, 10, cnt+1)
    Z = data.reshape(dimY,dimX)
    Z = Z[::-1,:]# flip vertical
    plt.xlim(0,dimX)
    plt.ylim(0,dimY)
    plt.pcolor(Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

def disp(data):
    plt.figure(figsize=(15,15))
    for i in range(100):
        draw_image(data[i][0],i)
    plt.show()
    
train, test = chainer.datasets.get_mnist()
disp(train)
