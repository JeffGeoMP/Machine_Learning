from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("D:\Repositorios\Machine_Learning\\files\\nac.csv")

#Grado del polinomio
polyDegree = 3

#Columnas a utilizar
X = df["Ano"].values.reshape((-1,1))
Y = df["Escuintla"]


polynomial = PolynomialFeatures(degree=polyDegree)
X_POLYNOMIAL = polynomial.fit_transform(X)          #Transformamos x para una funcion polinomial

#Entremamos Modelo
model = LinearRegression().fit(X_POLYNOMIAL, Y)
Y_POLYNOMIAL = model.predict(X_POLYNOMIAL)


#Errores en Prediccion
errorMean = round(mean_squared_error(Y, Y_POLYNOMIAL, squared=True) , 4)
r2 =  round(r2_score(Y, Y_POLYNOMIAL) , 4)

print("Coeficientes: ", model.coef_)
print("Intercept", model.intercept_)
print("Error medio: ",errorMean)
print("R2: ", r2)



fig, ax = plt.subplots()
ax.plot(X, Y_POLYNOMIAL)
ax.scatter(X,Y)
plt.show()

prediccion = 2050
x_new_min = 0.0
x_new_max = float(prediccion)  #para el calculo de la prediccion

X_NEW = np.linspace(x_new_min,x_new_max,50)
X_NEW = X_NEW[:,np.newaxis]

X_NEW_TRANSF = polynomial.fit_transform(X_NEW)
Y_NEW = model.predict(X_NEW_TRANSF)

print(Y_NEW[Y_NEW.size-1])