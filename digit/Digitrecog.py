# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:07:23 2017

@author: hongwfan
"""
import csv
import numpy as np
from os import listdir
def loadTrainData():
    l=[]
    with open('D:\\Kagggle\\digit\\train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=np.array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def toInt(array):
    array=np.mat(array)
    m,n=np.shape(array)
    newArray=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
                newArray[i,j]=int(array[i,j])
    return newArray

def nomalizing(array):
    m,n=np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTestData():  
    l=[]  
    with open('D:\\Kagggle\\digit\\test.csv') as file:  
         lines=csv.reader(file)  
         for line in lines:  
             l.append(line)  #28001*784  
    l.remove(l[0])  
    data=np.array(l)  
    return nomalizing(toInt(data)) 

#def loadTestResult():  
    #l=[]  
    #with open('D:\\Kagggle\\digit\\knn_benchmark.csv') as file:  
         #lines=csv.reader(file)  
         #for line in lines:  
             #l.append(line)  
     #28001*2  
    #l.remove(l[0])  
    #label=np.array(l)  
    #return toInt(label[:,1]) 
    
def classify(inX, dataSet, labels, k):  
    inX=np.mat(inX)  
    dataSet=np.mat(dataSet)  
    labels=np.mat(labels)  
    dataSetSize = dataSet.shape[0]                    
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet     
    sqDiffMat = np.array(diffMat)**2  
    sqDistances = sqDiffMat.sum(axis=1)                    
    distances = sqDistances**0.5  
    sortedDistIndicies = distances.argsort()              
    classCount={}                                        
    for i in range(k):  
        voteIlabel = labels[0,sortedDistIndicies[i]]  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  
    return sortedClassCount[0][0]