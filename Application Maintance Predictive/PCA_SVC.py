# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:54:35 2022

@author: CRISTIAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


if __name__ == "__main__":
    
    df = pd.read_csv('ai4i2020.csv')
    print(df.head)
    print('='*64)

   # df.info()
    corr = df.set_index('UDI').corr()
    sm.graphics.plot_corr(corr, xnames=list(corr.columns))
    plt.show()
    X = df.drop(['UDI','Product ID','Type',"Machine failure",'TWF','HDF','PWF','OSF','RNF'], axis=1)
    y = np.array(df["Machine failure"])
    class_names = list(X.columns)

    
    scaler = StandardScaler()
    segmentacion = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(segmentacion)
    
    plt.figure(figsize= (10,8))
    plt.plot(range(1,6), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    
    pca = PCA(n_components=3)
    pca.fit(segmentacion)
    scores_pca = pca.transform(segmentacion)
    
    X_train, X_test, y_train, y_test = train_test_split(scores_pca,y,test_size=0.35)
    
   
    
    # Creamos el clasificador
    classifier =  SVC()
    
    # Realizamos el entrenamiento
    classifier.fit(X_train, y_train)
    
    
    # Realizamos prediccion con nuestro conjunto de test

    predicted = classifier.predict(X_test)

    print(
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
      )
    print(f'Precisión {round(classifier.score(X_test, y_test),4)} %')

    print('='*64)

    plt.figure(figsize=(20,5))
    plt.plot(y_test,'--',predicted,'--')
    plt.title('Clases seleccionadas vs clases de prediccion')
    
    print('Matriz de Confusion')
    titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
                      ]   
    
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
    )

    print(disp.confusion_matrix)

    plt.show()