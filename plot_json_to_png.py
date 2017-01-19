import numpy as np
import matplotlib.pyplot as plt
import json
import os
import six


root = './groupA/results/'
for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' main/accuracy:')
    im = []
    for i in range(len(jsonData)):
        #print(jsonData[i][u'main/accuracy'])
        im.append(jsonData[i][u'main/accuracy'])
    #print(im)
    plt.plot(np.array(im))

plt.title(r"all main/accuracy")
plt.savefig('main_accuracy.png')
plt.show()

for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' main/loss:')
    im = []
    for i in range(len(jsonData)):
        #print(jsonData[i][u'main/loss'])
        im.append(jsonData[i][u'main/loss'])
    #print(im)
    plt.plot(np.array(im))
        
plt.title(r"all main/loss")
plt.savefig('main_loss.png')
plt.show()

for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' validation/main/accuracy:')
    im = []
    for i in range(len(jsonData)):
        #print(jsonData[i][u'validation/main/accuracy'])
        im.append(jsonData[i][u'validation/main/accuracy'])
    #print(im)
    plt.plot(np.array(im))

plt.title(r"all validation/main/accuracy")
plt.savefig('validation_main_accuracy.png')
plt.show()

for u in os.listdir(root):
    ##print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    ##print json.dumps(jsonData, sort_keys = True, indent = 4)
    #print(u + ' validation/main/loss:')
    im = []
    for i in range(len(jsonData)):
        #print(jsonData[i][u'validation/main/loss'])
        im.append(jsonData[i][u'validation/main/loss'])
    #print(im)
    plt.plot(np.array(im))
        
plt.title(r"all validation/main/loss")
plt.savefig('validation_main_loss.png')
plt.show()

f.close()
