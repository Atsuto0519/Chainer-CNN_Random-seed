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

hoge = 0
hogehoge = 0

root_A = './groupA/results_SGD/'
root_B = './groupB/results_SGD/'
dir_count = 0
im_A = [[],[],[],[]]
im_B = [[],[],[],[]]

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


    hogehoge = 0
    plt.title(r"main/accuracy")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    for i in range(len(im_A[0])):
        plt.plot(im_A[0][i], color=cm.gray(float(i)/len(im_A[0])))
        hoge = 0
        for j in range(len(im_A[0][i])):
            hoge += im_A[0][i][j]
        for j in range(len(im_B[0][i])):
            hoge += im_B[0][i][j]
        if (hoge > hogehoge):
            best_main_accuracy = i
            hogehoge = hoge
    plt.savefig('A_main_accuracy.png')
    plt.show()
    print(best_main_accuracy)

    hogehoge = 100
    plt.title(r"main/loss")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    for i in range(len(im_A[1])):
        plt.plot(im_A[1][i], color=cm.gray(float(i)/len(im_A[1])))
        hoge = 0
        for j in range(len(im_A[1][i])):
            hoge += im_A[1][i][j]
        for j in range(len(im_A[1][i])):
            hoge += im_B[1][i][j]
        if (hoge < hogehoge):
            best_main_loss = i
            hogehoge = hoge
    plt.savefig('A_main_loss.png')
    plt.show()
    print(best_main_loss)

    hogehoge = 0
    plt.title(r"validation/main/accuracy")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    for i in range(len(im_A[2])):
        plt.plot(im_A[2][i], color=cm.gray(float(i)/len(im_A[2])))
        hoge = 0
        for j in range(len(im_A[2][i])):
            hoge += im_A[2][i][j]
        for j in range(len(im_B[2][i])):
            hoge += im_B[2][i][j]
        if (hoge > hogehoge):
            best_validation_main_accuracy = i
            hogehoge = hoge
    plt.savefig('A_validation_main_accuracy.png')
    plt.show()
    print(best_validation_main_accuracy)

    hogehoge = 100
    plt.title(r"all validation/main/loss")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    for i in range(len(im_A[3])):
        plt.plot(im_A[3][i], color=cm.gray(float(i)/len(im_A[3])))
        hoge = 0
        for j in range(len(im_A[3][i])):
            hoge += im_A[3][i][j]
        for j in range(len(im_B[3][i])):
            hoge += im_B[3][i][j]
        if (hoge < hogehoge):
            best_validation_main_loss = i
            hogehoge = hoge
    plt.savefig('A_validation_main_loss.png')
    plt.show()
    print(best_validation_main_loss)
    
    plt.title(r"all best parameters")
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.grid(linestyle='--')
    plt.plot(im_A[0][best_main_accuracy], color=cm.gray(float(0)/4), label='gropuA/main/accuracy')
    plt.plot(im_A[1][best_main_loss], color=cm.gray(float(1)/4), label='groupA/main/loss')
    plt.plot(im_A[2][best_validation_main_accuracy], color=cm.gray(float(2)/4), label='groupA/validation/main/accuracy')
    plt.plot(im_A[3][best_validation_main_loss], color=cm.gray(float(3)/4), label='groupA/validation/main/loss')
    plt.plot(im_B[0][best_main_accuracy], color=cm.gray(float(0)/4), label='gropuB/main/accuracy')
    plt.plot(im_B[1][best_main_loss], color=cm.gray(float(1)/4), label='groupB/main/loss')
    plt.plot(im_B[2][best_validation_main_accuracy], color=cm.gray(float(2)/4), label='groupB/validation/main/accuracy')
    plt.plot(im_B[3][best_validation_main_loss], color=cm.gray(float(3)/4), label='groupB/validation/main/loss')
    plt.legend(loc='upper right', fontsize=10.5)
    plt.savefig('best_parameter.png')
    plt.show()
    
