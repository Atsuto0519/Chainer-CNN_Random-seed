import chainer
from chainer import training

train, test = chainer.datasets.get_mnist()

print(len(train))
print(len(train[0]))
print("train[0][0] =", train[0][0])
print("train[0][1] =", train[0][1])
print()

for i in range(10):
    print("train[%d][0] = %d" % (i, int(train[i][1])))
print()



import matplotlib.pyplot as plt


def draw_image_onlyone(data):
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


index = 7
print("label=%d" % (train[index][1]))
draw_image_onlyone(train[index][0])
plt.show()

