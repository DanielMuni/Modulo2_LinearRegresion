#Daniel Munive Meneses
#A01734205

#Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
#Regresion Linear

#Se importan las librerias necesarias del proyecto

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




#Se importa la base de datos
data = pd.read_csv('./IceCreamData.csv')

#Se selecciono las columas con las que voy a trabajar
X = np.array(data.iloc[:,0]).reshape(-1,1) #Celsius
Y = np.array(data.iloc[:,1]).reshape(-1,1) #Revenue (Dolares)


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.25)

regr = LinearRegression(fit_intercept =False)
regr.fit(x_train,y_train)
print(regr.score(x_test,y_test), "\n")

print(regr.coef_)

print("--------------------------------------------------------------------------------")

#Se calculan las predicciones para la regresion linear
print(regr.predict([[50],[30],[25],[0], [-20]]))


#Creamos una grafica con los resultados
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(y_test),max(y_test)], color='red')
plt.xlabel("Temperature(Celsius)")
plt.ylabel("Revenue(Dolars)")
plt.title("Temperature vs Revenue")
plt.show()





