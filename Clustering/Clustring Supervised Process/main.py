# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:47:29 2022

@author: CRISTIAN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
from fcmeans import FCM
import pandas as pd

#datos sin normalizar
data = pd.read_csv('base_multi.csv',delimiter=';', header=None)
t=np.arange(1,1632,1)

data[1] = data[1].apply(lambda x: float(x.split()[0].replace(',', '.')))
data[2] = data[2].apply(lambda x: float(x.split()[0].replace(',', '.')))


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

segmentacion = scaler.fit_transform(data)

plt.figure(figsize=(20,5))
#graficacion de datos normalizados
plt.plot(t,segmentacion[:,0],t,segmentacion[:,1],t,segmentacion[:,2],t,segmentacion[:,3])

# #para ordenar la base de datos en un array de 4 columnas con los datos 
# #normalizados
# #datos=np.array([normsp,normy,data[:,2],normq]).transpose()
# datos=np.array([normsp,normy,normq]).transpose()
# #Uso del clasificador FCMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch

clustering = Birch().fit(segmentacion)


c_labels = clustering.fit_predict(segmentacion)


#Creo una lista correspondiente a 10 colores que me van a identificar cada clase
colores=['blue','green','red','cyan','magenta','yellow','black','white','orange','brown']
asignar=[]
#un ciclo for para que a cada dato que se le hizo la prediccion le asigne un color
for row in c_labels:
      asignar.append(colores[row])
    
# #Se crea un arreglo para añadir las diferentes clases que se predijo
labels2=np.array(c_labels)#graficacion de las clases 
plt.figure(figsize=(20,5))
labels2=np.array(c_labels)
plt.plot(t,labels2)

#Graficiacion de la base de datos con su parte clasificada
plt.figure(figsize=(20,5))
plt.scatter(t,segmentacion[:,0],c=asignar,s=30)
plt.scatter(t,segmentacion[:,1],c=asignar,s=30)
plt.scatter(t,segmentacion[:,2],c=asignar,s=30)
plt.scatter(t,segmentacion[:,3],c=asignar,s=30)

# plt.figure(figsize=(20,5))
# grados_pertenencia=fcm.u
# plt.plot(grados_pertenencia)