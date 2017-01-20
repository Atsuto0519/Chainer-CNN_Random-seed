import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os
import six

A_best_main_accuracy = 0
A_best_main_loss = 0
A_best_validation_main_accuracy = 0
A_best_validation_main_loss = 0
B_best_main_accuracy = 0
B_best_main_loss = 0
B_best_validation_main_accuracy = 0
B_best_validation_main_loss = 0

hoge = 0
hogehoge = 0

root_A = './groupA/results/'
root_B = './groupB/results/'
dir_count = 0
im_A = [[],[],[],[]]
im_B = [[],[],[],[]]
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
    dir_count += 1

dir_count = 0
for u in os.listdir(root_B):
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
    
hogehoge = 100
plt.title(r"main/accuracy")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_A[0])):
    plt.plot(im_A[0][i], color=cm.gray(float(i)/len(im_A[0])))
    hoge = 0
    for j in range(len(im_A[0][i])):
        hoge += im_A[0][i][j]
    if (hoge < hogehoge):
        A_best_main_accuracy = i
        hogehoge = hoge
plt.savefig('A_main_accuracy.png')
plt.show()
print(A_best_main_accuracy)

hogehoge = 0
plt.title(r"main/loss")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_A[1])):
    plt.plot(im_A[1][i], color=cm.gray(float(i)/len(im_A[1])))
    hoge = 0
    for j in range(len(im_A[1][i])):
        hoge += im_A[1][i][j]
    if (hoge > hogehoge):
        A_best_main_loss = i
        hogehoge = hoge
plt.savefig('A_main_loss.png')
plt.show()
print(A_best_main_loss)

hogehoge = 100
plt.title(r"validation/main/accuracy")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_A[2])):
    plt.plot(im_A[2][i], color=cm.gray(float(i)/len(im_A[2])))
    hoge = 0
    for j in range(len(im_A[2][i])):
        hoge += im_A[2][i][j]
    if (hoge < hogehoge):
        A_best_validation_main_accuracy = i
        hogehoge = hoge
plt.savefig('A_validation_main_accuracy.png')
plt.show()
print(A_best_validation_main_accuracy)

hogehoge = 0
plt.title(r"all validation/main/loss")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_A[3])):
    plt.plot(im_A[3][i], color=cm.gray(float(i)/len(im_A[3])))
    hoge = 0
    for j in range(len(im_A[3][i])):
        hoge += im_A[3][i][j]
    if (hoge > hogehoge):
        A_best_validation_main_loss = i
        hogehoge = hoge
plt.savefig('A_validation_main_loss.png')
plt.show()
print(A_best_validation_main_loss)

plt.title(r"all best parameters")
plt.xlabel('iteration')
plt.ylabel('value')
plt.plot(im_A[0][A_best_main_accuracy], color=cm.gray(float(0)/4), label='main/accuracy')
plt.plot(im_A[1][A_best_main_loss], color=cm.gray(float(1)/4), label='main/loss')
plt.plot(im_A[2][A_best_validation_main_accuracy], color=cm.gray(float(2)/4), label='validation/main/accuracy')
plt.plot(im_A[3][A_best_validation_main_loss], color=cm.gray(float(3)/4), label='validation/main/loss')
plt.legend(loc='upper right')
plt.savefig('A_best_parameter.png')
plt.show()

hogehoge = 100
plt.title(r"main/accuracy")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_B[0])):
    plt.plot(im_B[0][i], color=cm.gray(float(i)/len(im_B[0])))
    hoge = 0
    for j in range(len(im_B[0][i])):
        hoge += im_B[0][i][j]
    if (hoge < hogehoge):
        B_best_main_accuracy = i
        hogehoge = hoge
plt.savefig('B_main_accuracy.png')
plt.show()
print(B_best_main_accuracy)

hogehoge = 0
plt.title(r"main/loss")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_B[1])):
    plt.plot(im_B[1][i], color=cm.gray(float(i)/len(im_B[1])))
    hoge = 0
    for j in range(len(im_B[1][i])):
        hoge += im_B[1][i][j]
    if (hoge > hogehoge):
        B_best_main_loss = i
        hogehoge = hoge
plt.savefig('B_main_loss.png')
plt.show()
print(B_best_main_loss)

hogehoge = 100
plt.title(r"validation/main/accuracy")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_B[2])):
    plt.plot(im_B[2][i], color=cm.gray(float(i)/len(im_B[2])))
    hoge = 0
    for j in range(len(im_B[2][i])):
        hoge += im_B[2][i][j]
    if (hoge < hogehoge):
        B_best_validation_main_accuracy = i
        hogehoge = hoge
plt.savefig('B_validation_main_accuracy.png')
plt.show()
print(B_best_validation_main_accuracy)

hogehoge = 0
plt.title(r"all validation/main/loss")
plt.xlabel('iteration')
plt.ylabel('value')
for i in range(len(im_B[3])):
    plt.plot(im_B[3][i], color=cm.gray(float(i)/len(im_B[3])))
    hoge = 0
    for j in range(len(im_B[3][i])):
        hoge += im_B[3][i][j]
    if (hoge > hogehoge):
        B_best_validation_main_loss = i
        hogehoge = hoge
plt.savefig('B_validation_main_loss.png')
plt.show()
print(B_best_validation_main_loss)

plt.title(r"all best parameters")
plt.xlabel('iteration')
plt.ylabel('value')
plt.plot(im_B[0][B_best_main_accuracy], color=cm.gray(float(0)/4), label='main/accuracy')
plt.plot(im_B[1][B_best_main_loss], color=cm.gray(float(1)/4), label='main/loss')
plt.plot(im_B[2][B_best_validation_main_accuracy], color=cm.gray(float(2)/4), label='validation/main/accuracy')
plt.plot(im_B[3][B_best_validation_main_loss], color=cm.gray(float(3)/4), label='validation/main/loss')
plt.legend(loc='upper right')
plt.savefig('B_best_parameter.png')
plt.show()

average_value = [[],[],[],[]]
for i in range(9):
    average_value[0].append((im_A[0][A_best_main_accuracy][i]+im_B[0][B_best_main_accuracy][i])/2)
    average_value[1].append((im_A[1][A_best_main_loss][i]+im_B[1][B_best_main_loss][i])/2)
    average_value[2].append((im_A[2][A_best_validation_main_accuracy][i]+im_B[2][B_best_validation_main_accuracy][i])/2)
    average_value[3].append((im_A[3][A_best_validation_main_accuracy][i]+im_B[3][B_best_validation_main_accuracy][i])/2)
    
plt.title(r"all average parameters")
plt.xlabel('iteration')
plt.ylabel('value')
plt.plot(average_value[0], color=cm.gray(float(0)/4), label='main/accuracy')
plt.plot(average_value[1], color=cm.gray(float(1)/4), label='main/loss')
plt.plot(average_value[2], color=cm.gray(float(2)/4), label='validation/main/accuracy')
plt.plot(average_value[3], color=cm.gray(float(3)/4), label='validation/main/loss')
plt.legend(loc='upper right')
plt.savefig('Average_A_and_B_parameter.png')
plt.show()

f.close()
