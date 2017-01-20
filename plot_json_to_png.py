import numpy as np
import matplotlib.pyplot as plt
import json
import os
import six

root = './groupA/results/'
root_A = './groupA/results/'
root_B = './groupB/results/'
dir_count=0
im_A = [[],[],[],[]]
im_B = [[]]
im = []
for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' main/accuracy:')
    im.append([])
    for i in range(4):
        im_A[i].append([])
    for i in range(len(jsonData)):
        #print(jsonData[i][u'main/accuracy'])
        im[dir_count].append(jsonData[i][u'main/accuracy'])
        im_A[0][dir_count].append(jsonData[i][u'main/accuracy'])
        im_A[1][dir_count].append(jsonData[i][u'main/loss'])
        im_A[2][dir_count].append(jsonData[i][u'validation/main/accuracy'])
        im_A[3][dir_count].append(jsonData[i][u'validation/main/loss'])
    #print(im)
    plt.plot(np.array(im[dir_count]))
    dir_count+=1

plt.title(r"all main/accuracy")
plt.savefig('main_accuracy.png')
plt.show()

for i in range(len(im_A[0])):
    plt.plot(im_A[0][i])
plt.show()

dir_count=0
im = []
for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' main/loss:')
    im.append([])
    for i in range(len(jsonData)):
        #print(jsonData[i][u'main/loss'])
        im[dir_count].append(jsonData[i][u'main/loss'])
    #print(im)
    plt.plot(np.array(im[dir_count]))
    dir_count+=1
        
plt.title(r"all main/loss")
plt.savefig('main_loss.png')
plt.show()

dir_count=0
im = []
for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' validation/main/accuracy:')
    im.append([])
    for i in range(len(jsonData)):
        #print(jsonData[i][u'validation/main/accuracy'])
        im[dir_count].append(jsonData[i][u'validation/main/accuracy'])
    #print(im)
    plt.plot(np.array(im[dir_count]))
    dir_count+=1

plt.title(r"all validation/main/accuracy")
plt.savefig('validation_main_accuracy.png')
plt.show()

dir_count=0
im = []
for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' validation/main/loss:')
    im.append([])
    for i in range(len(jsonData)):
        #print(jsonData[i][u'validation/main/loss'])
        im[dir_count].append(jsonData[i][u'validation/main/loss'])
    #print(im)
    plt.plot(np.array(im[dir_count]))
    dir_count+=1
        
plt.title(r"all validation/main/loss")
plt.savefig('validation_main_loss.png')
plt.show()

f.close()
