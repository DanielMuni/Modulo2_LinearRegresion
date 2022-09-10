#Daniel Munive Meneses
#A01734205

#Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
#Regresion Linear

#Se importan las librerias necesarias del proyecto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
Pred_error_train = Y_train - Y_pred_train

print("MSE: ",mean_squared_error(Y_test, Y_pred))
print("Model score: ", regr.score(X_test, Y_test))

#Plot de la regresión
figure, axis = plt.subplots(2,3)

#TEST
axis[0,0].scatter(X_test, Y_test)
axis[0,0].plot(X_test, Y_pred, color='red')
axis[0,0].set_title("Temperature(Celsius) vs Revenue(Dolars) (test data)")
axis[0,0].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')

#histograma(bias)
axis[0,1].hist(Pred_error_test)
axis[0,1].set_title('Histogram of test prediction error')
axis[0,1].set_xlim(-200, 200)
axis[0,1].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')

#varianza(?)
axis[0,2].scatter(X_test, Y_test, alpha = 0.3, label = 'Real data')
axis[0,2].scatter(X_test, Pred_error_test, color='orange',alpha = 0.1, label = 'Predicted data')
axis[0,2].set_title("Real test data vs Predicted test data")
axis[0,2].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')
axis[0,2].legend()


#TRAIN
axis[1,0].scatter(X_train, Y_train)
axis[1,0].plot(X_train, Y_pred_train, color ='red')
axis[1,0].set_title("Temperature(Celsius) vs Revenue(Dolars) (train data)")
axis[1,0].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')

#histograma(bias)
axis[1,1].hist(Pred_error_train)
axis[1,1].set_title('Histogram of train prediction error')
axis[1,1].set_xlim(-200, 200)
axis[1,1].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')

#varianza(?)
axis[1,2].scatter(X_train, Y_train, alpha = 0.3, label = 'Real data')
axis[1,2].scatter(X_train, Pred_error_train, color='orange',alpha = 0.1, label = 'Predicted data')
axis[1,2].set_title("Real train data vs Predicted train data")
axis[1,2].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')
axis[1,2].legend()
plt.show()
plt.show()



