import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os
import six

best_main_accuracy = 0
best_main_loss = 0
best_validation_main_accuracy = 0
best_validation_main_loss = 0
thereshold = 0.90
best_seed = 0
min_iteration = 0

root_A = './groupA/results_SGD/'
root_B = './groupB/results_SGD/'
dir_count = 0
im_A = [[],[],[],[]]
im_B = [[],[],[],[]]
im_C = []

if os.listdir(root_A)==os.listdir(root_B):
    for u in os.listdir(root_A):
        f = open(root_A + u + '/log', 'r')
        jsonData = json.load(f)
        for i in range(4):
            im_A[i].append([])
        for i in range(len(jsonData)):
            im_A[0][dir_count].append(jsonData[i]['main/accuracy'])
            im_A[1][dir_count].append(jsonData[i]['main/loss'])
            im_A[2][dir_count].append(jsonData[i]['validation/main/accuracy'])
            im_A[3][dir_count].append(jsonData[i]['validation/main/loss'])

        f = open(root_B + u + '/log', 'r')
        jsonData = json.load(f)
        for i in range(4):
            im_B[i].append([])
        for i in range(len(jsonData)):
            im_B[0][dir_count].append(jsonData[i]['main/accuracy'])
            im_B[1][dir_count].append(jsonData[i]['main/loss'])
            im_B[2][dir_count].append(jsonData[i]['validation/main/accuracy'])
            im_B[3][dir_count].append(jsonData[i]['validation/main/loss'])
        dir_count += 1

    dir_count=0
    min_iteration=30
    plt.title(r"testdata's accuracy")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    for i in range(len(im_A[2])):
        im_C.append([])
        for j in range(len(im_A[2][i])):
            im_C[dir_count].append((im_A[2][i][j]+im_B[2][i][j])/2)

            if im_C[dir_count][j]>=thereshold:
                if min_iteration>j:
                    min_iteration = j
                    best_seed = dir_count

        
        plt.plot(im_C[dir_count], color=cm.gray(float(i)/len(im_A[2])))
        print min_iteration
        print best_seed
        print dir_count
        print 
        dir_count+=1

    plt.savefig('all_testdata_accuracy.png')
    plt.show()


    print(min_iteration)
    print(best_seed)
    plt.title(r"best testdata's accuracy")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    plt.plot(im_C[best_seed], color=cm.gray(1))
    plt.savefig('best_testdata_accuracy_'+str(thereshold)+'_.png')
    plt.show()
