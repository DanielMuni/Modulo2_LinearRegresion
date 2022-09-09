#Daniel Munive Meneses
#A01734205

#Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
#Regresion Linear

#Se importan las librerias necesarias del proyecto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Obtención de los datos a partir de un csv con pandas
columns = ["Temperature", "Revenue"]
df = pd.read_csv('IceCreamData.csv', names = columns)

#Me aseguro que los datos que voy a ocupar esten el el tipo de dato correcto, para poder trabajar con ellos
df['Temperature']= df['Temperature'][1:].astype(float)
df['Revenue'] = df['Revenue'][1:].astype(float)

#Limpieza de datos, considerando que no puede haber datos vacíos en dichas columnas
df = df.drop(df[df.Temperature.isnull()].index)
df = df.drop(df[df.Revenue.isnull()].index)



#Los usuarios deben haber visto el anime por lo menos una vez
df = df[df['Revenue'] > 0]

print(df)


#eterminación de la variable correspondiente al 'eje X' y 'eje Y'
X = np.array(df['Temperature']).reshape(-1,1)
Y = np.array(df['Revenue']).reshape(-1,1)

#Division del data set en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)

#Se hace el calculo del la regresion por medio de los metoddos de sklearn
regr = LinearRegression(fit_intercept = False).fit(X_train,Y_train)

#Coeficiente e Intercepto
print("Coefficient: ",regr.coef_)
print("Intercept: ",regr.intercept_)

#Predicciones
print("Predicitons:")
custom_pred = [[50.0], [30.0], [25.0], [0.0], [-20.0]]

for i in custom_pred:
    print(f"Temperatura en Celsius: {i} Ganancia(Dolares): {regr.predict([i])}")

#Obtencion del error de prediccion en test y train
Y_pred = regr.predict(X_test)
Pred_error_test = Y_pred - Y_test
Y_pred_train = regr.predict(X_train)
Pred_error_train = Y_pred_train - Y_train

#Plot de la regresión
figure, axis = plt.subplots(2,2)

axis[0,0].scatter(X_test, Y_test)
axis[0,0].plot(X_test, Y_pred, color='red')
axis[0,0].set_title("Temperature(Celsius) vs Revenue(Dolars) (test data)")

axis[0,1].hist(Pred_error_test)
axis[0,1].set_title('Histogram of test prediction error')
axis[0,1].set_xlim(-400, 400)

axis[1,0].scatter(X_train, Y_train)
axis[1,0].plot(X_train, Y_pred_train, color ='red')
axis[1,0].set_title("Temperature(Celsius) vs Revenue(Dolars) (train data)")

axis[1,1].hist(Pred_error_train)
axis[1,1].set_title('Histogram of train prediction error')
axis[1,1].set_xlim(-400, 400)

plt.show()