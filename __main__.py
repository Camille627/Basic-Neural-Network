# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 22:24:11 2024

from youtube : https://www.youtube.com/watch?v=bzC5cdxZcOM&t=446s

@author: Camille
"""

import numpy as np

x_enter = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[1,1.5]),dtype=float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype = float) # données de sortie 1 = Rouge / 0 = Bleu

x_enter = x_enter/np.amax(x_enter,axis=0)

X = np.split(x_enter, [8])[0]
xPrediction = np.split(x_enter, [8])[1]

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2 # nombre synapse(s) d'entrée
        self.outputSize = 1 # nombre synapse(s) de sortie
        self.hiddenSize = 3 # nobre de synapse(s) couche cachée
        
        # On génère les poids aléatoirement uniquement à la création
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        
    def forward(self,X): # prend les entrees et retourne la sortie prédite par l'IA
        self.z = np.dot(X,self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2,self.W2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    
    def sigmoidPrime(self,s): # derivee de sigmoid
        return s*(1-s)
    
    def backward(self,X,y,o) :
        # Fonction de retropropagation dont le but est de corriger l'erreur de l'IA
        # X:input , y:expected_output , o:forward/real_output
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
    def train(self,X,y):
        o = self.forward(X)
        self.backward(X, y, o)
        
    def predict(self):
        print("Donnee predite apres entrainement")
        print("Entree : \n"+str(xPrediction))
        print("Sortie : \n"+str(self.forward(xPrediction)))
        
        if(self.forward(xPrediction)<0.5):
            print("La fleur est bleue. \n")
        else:
            print("La fleur est rouge. \n")
        
        
    
NN = Neural_Network()

n = 1000000 #nombre d'iteration de l'entrainement
for i in range(n):
    if(i==n-1):
        print("# "+str(i)+"\n")
        print("Valeurs d'entree: \n"+str(X))
        print("Sortie attendue: \n"+str(y))
        print("Sortie predite: \n"+str(np.matrix.round(NN.forward(X),2)))
        print("\n")
    NN.train(X, y)

NN.predict()