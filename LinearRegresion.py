#Daniel Munive Meneses
#A01734205

#Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
#Regresion Linear

#Se importan las librerias necesarias del proyecto
import pandas as pd
import matplotlib.pyplot as plt

#Se importa la base de datos
data = pd.read_csv('./IceCreamData.csv')

#Se selecciono las columas con las que voy a trabajar
X = data.iloc[:,0].astype(float) #Celsius
Y = data.iloc[:,1].astype(float) #Revenue (Dolares)

# se definen las variables que vamos a ocupar para la regresion linear
m = 0
c = 0
L = .0001
i = 10000

#Para aplicar de manera correcta el Machine Learning calcularemos la regresion Linear con Gradiente Descendiente

def GradientDesc(m,c,alpha,iterations,x,y):

	iter = 0 #Variable temporal para las iteraciones
	n = float(len(x)) #Variable para guardar el promedio

	while True: 
		Y_pred = m * X + c
		D_m = (-2/n) * sum(x * ( y - Y_pred )) #Derivada de Dm
		D_c = (-2/n) * sum( y - Y_pred) #Derivada de Dc
		m = m -alpha*D_m
		c = c -alpha*D_c
		iter = iter + 1
		if( iter == iterations ):
			print("m: ", m)
			print("c: ", c)
			break
	
	return(m,c)

#Se calculan las predicciones para la regresion linear
def Prediction(m,c,x_input):
    y = m*x_input+c
    print(f"Temperatura en Celsius {x_input} Ganancia(Dolares) {y}")
	

    return

m, c = GradientDesc(m,c,L,i,X,Y)

y_pred = m*X + c


#Predicciones
Prediction(m,c,50.0)
Prediction(m,c,30.0)
Prediction(m,c,25.0)
Prediction(m,c,0.0)
Prediction(m,c,-20.0)




#Creamos una grafica con los resultados
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(y_pred),max(y_pred)], color='red')
plt.xlabel("Temperature(Celsius)")
plt.ylabel("Revenue(Dolars)")
plt.title("Temperature vs Revenue")
plt.show()