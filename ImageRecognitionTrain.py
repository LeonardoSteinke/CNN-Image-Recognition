# -*- coding: utf-8 -*-
"""
Created on Sat Mar  27 14:48:22 2021

@author: Leonardo Tadeu Jaques Steinke
"""

Epocas = 50
totalCachorros = 7000
totalGatos = 7000

colunasExibicao = 5

import cv2 
import numpy as np
import matplotlib.pyplot as plt

import os #Sistema Operacional
import random
import gc #Garbage Collector

dirTreino = 'C:\\CNNImageRecognition\\treino'
dirTeste = 'C:\\CNNImageRecognition\\teste'

treina_dog = ['C:\\CNNImageRecognition\\treino\\{}'.format(i) for i in os.listdir(dirTreino) if 'dog' in i] #Encontra as imagens de cachorros
treina_cat = ['C:\\CNNImageRecognition\\treino\\{}'.format(i) for i in os.listdir(dirTreino) if 'cat' in i] #Encontra as imagens de gatos

teste_imgs = ['C:\\CNNImageRecognition\\teste\\{}'.format(i) for i in os.listdir(dirTeste)] #Encontra o diretorio de teste

treino_img = treina_dog[:totalCachorros] + treina_cat[:totalGatos]
random.shuffle(treino_img) #Embaralha

#Limpa coisas desnecessarias
del treina_dog
del treina_cat
gc.collect()

#Redimencionar as imagens
imgX = 150
imgY = 150
channels = 3 #Trocar para 1 caso queira imagens preto e branco

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
    

X, y = readAndProcessImg(treino_img);

import seaborn as sns
del treino_img
gc.collect();

#Converte lista em um array numpy
X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Rótulos para Gatos e Cachorros:')

print("Formato das imagens de treino: ", X.shape)
print("Formato dos rótulos          :", y.shape)

from sklearn.model_selection import train_test_split
X_treino, X_val, y_treino, y_val = train_test_split(X, y, test_size=0.20,  random_state=2)

print("Formato imagens de treino:", X_treino.shape)
print("Formato imagens de validação: ", X_val.shape)
print("Formato label: ", y_treino.shape)
print("Formato label: ", y_val.shape)

del X
del y
gc.collect()

nTreino = len(X_treino)
nval = len(X_val)

batch_size = 32


from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary();

#Sera utilizado RMSprop optimizer com uma taxa de aprendizado de 0.0001
#Sera utilizado binary_crossentropy loss porque é uma classificação binária
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


#Configuração de aumento para previnir overfitting, já que sera um dataset pequeno
treino_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)

#Gerador de Imgens
gerador_treino = treino_datagen.flow(X_treino, y_treino, batch_size=batch_size)
gerador_val = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# >>>>>>>>>>>>>>>>>>>>>>
# >>>>Inicio treinamento
# >>>>>>>>>>>>>>>>>>>>>>

#100 passos por epoca
history = model.fit(gerador_treino,
                              steps_per_epoch=nTreino // batch_size,
                              epochs=Epocas,
                              validation_data=gerador_val,
                              validation_steps=nval // batch_size)

model.save('modelo'+str(Epocas)+'epocas.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Precisão de treinamento e validação
plt.plot(epochs, acc, 'b', label='Precisão de treinamento')
plt.plot(epochs, val_acc, 'r', label='Precisão de validação')
plt.title('Precisão de treinamento e validação')
plt.legend()
plt.figure()
#Perda de treino e validação
plt.plot(epochs, loss, 'b', label='Perda de treino')
plt.plot(epochs, val_loss, 'r', label='Perda de validação')
plt.title('Perda de treino e validação')
plt.legend()
plt.show()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> FIM da Parte de Treinamento
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#%%

from keras.models import load_model
model=load_model('modelo'+str(Epocas)+'epocas.h5')
test_imgs = ['C:\\CNNImageRecognition\\teste\\{}'.format(i) for i in os.listdir(dirTeste)]

ImagensParaAvaliar = 12

#Now lets predict on the first ImagensParaAvaliar of the test set
X_teste, y_teste = readAndProcessImg(test_imgs[0:ImagensParaAvaliar]) #y_test in this case will be empty.
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

