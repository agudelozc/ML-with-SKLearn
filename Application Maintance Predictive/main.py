# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:06:52 2022

@author: CRISTIAN
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:54:35 2022

@author: CRISTIAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay



names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",

]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),

]


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
    X_train, X_test, y_train, y_test = train_test_split(segmentacion,y,test_size=0.4, random_state=42)
        
    scores = []
    for name, clf in zip(names, classifiers):
        
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        
    print(f"El mejor Score encontrado fue de {max(scores)}")
    print('='*64)
    print(f"Que corresponde al clasificador {classifiers[scores.index(max(scores))]}")
    print('='*64)
    
    classifier = classifiers[scores.index(max(scores))]
        

    # Realizamos prediccion con nuestro conjunto de test
    
    y_pred = classifier.predict(X_test)
    
    print(
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_test, y_pred)}\n"
      )
    print(f'Precisión {round(classifier.score(X_test, y_test),4)} %')
    
    print('='*64)
    
    
    """
    Confusion Matriz
    ------------------------------------------
    |Verdadero Negativo | Falsos Positivos  |
    -------------------------------------------
    |Falsos Negativos   |  Verdadero Positivo|
    ------------------------------------------
    """
    
    """
    Verdadero positivo: El valor real es positivo 
    y  la prueba predijo tambien que era positivo.
    
    Verdadero negativo: El valor real  es negativo
    y la prueba predijo tambien que era negativo.
    
    Falso negativo: El valor real es positivo, y 
    la prueba predijo  que el resultado es negativo. 
    
    Falso positivo: El valor real es negativo, y la 
    prueba predijo  que el resultado es positivo.
    """
    
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred)
    
    plt.show()