# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:02:18 2021

@author: Leonardo Tadeu Jaques Steinke
"""

import cv2 
import numpy as np
import os #Sistema Operacional
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

dirTeste = 'teste'

#Redimencionar as imagens
imgX = 150
imgY = 150

def readAndProcessImg(listImgs):
    X = []
    y = [] 

    for image in listImgs:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (imgX, imgY), interpolation=cv2.INTER_CUBIC))
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    
    return X, y
    
colunasExibicao = 5
Epocas = 50

model=load_model('modelo'+str(Epocas)+'epocas.h5')
test_imgs = ['teste\\{}'.format(i) for i in os.listdir(dirTeste)]

ImagensParaAvaliar = 20

X_teste, y_teste = readAndProcessImg(test_imgs[0:ImagensParaAvaliar])
x = np.array(X_teste)
test_datagen = ImageDataGenerator(rescale=1./255)
i = 0
text_labels = []

plt.figure(figsize=(20,20))

for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append(f'Cachorro {pred}')
    else:
        text_labels.append(f'Gato {pred}')
    #Número de linhas, número de colunas
    plt.subplot((ImagensParaAvaliar / colunasExibicao) + 1, colunasExibicao, i + 1)
    plt.title('' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % ImagensParaAvaliar == 0:
        break
plt.show()
