import numpy as np
import matplotlib.pyplot as plt
import json
import os
import six


root = './groupA/results/'
for u in os.listdir(root):
    #print(u)
    f = open(root + u + '/log', 'r')
    jsonData = json.load(f)
    #print json.dumps(jsonData, sort_keys = True, indent = 4)
    print(u + ' main/loss:')
    for i in range(len(jsonData)):
        print(jsonData[i][u'main/loss'])

f.close()
