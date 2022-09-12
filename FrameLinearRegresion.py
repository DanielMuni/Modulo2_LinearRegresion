#Daniel Munive Meneses
#A01734205

#Módulo 2 Implementación de una técnica de aprendizaje máquina con el uso de un framework.
#Regresion Linear

#Se importan las librerias necesarias del proyecto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso


#Obtención de los datos a partir de un csv con pandas
columns = ["Temperature", "Revenue"]
df = pd.read_csv('IceCreamData.csv', names = columns)

#Me aseguro que los datos que voy a ocupar esten el el tipo de dato correcto, para poder trabajar con ellos
df['Temperature']= df['Temperature'][1:].astype(float)
df['Revenue'] = df['Revenue'][1:].astype(float)

#Limpieza de datos, considerando que no puede haber datos vacíos en dichas columnas
df = df.drop(df[df.Temperature.isnull()].index)
df = df.drop(df[df.Revenue.isnull()].index)
df = df[df['Revenue'] > 0]

print(df)


#Determinación de la variable correspondiente al 'eje X' y 'eje Y'
X = np.array(df['Temperature']).reshape(-1,1)
Y = np.array(df['Revenue']).reshape(-1,1)

#Division del data set en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)

#Se hace el calculo del la regresion por medio de los metodos de sklearn
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

print("MSE test: ",mean_squared_error(Y_test, Y_pred))
print("Model score test: ", regr.score(X_test, Y_test))

print("MSE train: ",mean_squared_error(Y_train, Y_pred_train))
print("Model score train: ", regr.score(X_train, Y_train))

#Plot de la regresión
figure, axis = plt.subplots(2,3)

#TEST
axis[0,0].scatter(X_test, Y_test)
axis[0,0].plot(X_test, Y_pred, color='red', label = "MSE: " + str(mean_squared_error(Y_test, Y_pred)))
axis[0,0].set_title("Temperature(Celsius) vs Revenue(Dolars) (test data)")
axis[0,0].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')
axis[0,0].legend()

#histograma(bias)
axis[0,1].hist(Pred_error_test)
axis[0,1].set_title('Histogram of test prediction error')
axis[0,1].set_xlim(-200, 200)
axis[0,1].set(xlabel = 'Rating prediction error (Y_test - Y_pred)', ylabel = 'Frequency')

#varianza(?)
axis[0,2].scatter(X_test, Y_test, alpha = 0.5, label = 'Real data')
axis[0,2].scatter(X_test, Pred_error_test, color='orange',alpha = 0.1, label = 'Predicted data')
axis[0,2].set_title("Real test data vs Predicted test data")
axis[0,2].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')
axis[0,2].legend()


#TRAIN
axis[1,0].scatter(X_train, Y_train)
axis[1,0].plot(X_train, Y_pred_train, color ='red',label = "MSE: " + str(mean_squared_error(Y_train, Y_pred_train)))
axis[1,0].set_title("Temperature(Celsius) vs Revenue(Dolars) (train data)")
axis[1,0].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')
axis[1,0].legend()

#histograma(bias)
axis[1,1].hist(Pred_error_train)
axis[1,1].set_title('Histogram of train prediction error')
axis[1,1].set_xlim(-200, 200)
axis[1,1].set(xlabel = 'Rating prediction error (Y_train - Y_pred_train)', ylabel = 'Frequency')

#varianza(?)
axis[1,2].scatter(X_train, Y_train, alpha = .5, label = 'Real data')
axis[1,2].scatter(X_train, Pred_error_train, color='orange',alpha = 0.1, label = 'Predicted data')
axis[1,2].set_title("Real train data vs Predicted train data")
axis[1,2].set(xlabel = 'Temperature(Celsius)', ylabel = 'Revenue(Dolars)')
axis[1,2].legend()
plt.show()

model_lasso = Lasso(alpha = 0.01)
model_lasso.fit(X_train, Y_train)
pred_train_lasso = model_lasso.predict(X_train)

print("MSE in Lasso train: ", mean_squared_error(Y_train, pred_train_lasso))
print("Lasso score train: ", r2_score(Y_train, pred_train_lasso))

pred_test_lasso = model_lasso.predict(X_test)
print("MSE in Lasso test: ", mean_squared_error(Y_test, pred_test_lasso))
print("Lasso score test: ", r2_score(Y_test, pred_test_lasso))


