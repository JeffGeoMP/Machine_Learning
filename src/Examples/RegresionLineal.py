from statistics import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("D:\Repositorios\Machine_Learning\\files\\nac.csv")

#top = df.columns.values
#print(top)

X = df["Ano"].values.reshape((-1,1))
#Y = df["Solola"]
Y = df["Republica"]

LinearRegression = linear_model.LinearRegression()
model = LinearRegression.fit(X,Y)
Y_PREDICT = model.predict(X)


#Model y = mx + b
print("Interseccion: (b): ", model.intercept_)
print("Pendiente (m): ", model.coef_)

#Errores en Prediccion
errorMean = round(mean_squared_error(Y, Y_PREDICT, squared=True) , 4)
r2 =  round(r2_score(Y, Y_PREDICT) , 4)

print("Error medio: ",errorMean);   #Representa a la raiz de la distancia cuadrada promedio entre el valor real y el pronosticado
print("R2: ", r2);                  #Aptitud dle modelo entre 0 y 1 se considera un buen modelo despues de 0.80

#Grafica
#plt.xlabel("Año")
#plt.ylabel("Numero de Nacimientos")
#plt.scatter(X, Y);
#plt.plot(X, Y_PREDICT, color='red', label = "Error Medio: 200898.35\nR^2: 0.1782");
#plt.title("Nacimientos en Solola por Año")
#plt.legend()
#plt.show();

fig, ax = plt.subplots(1,1)  #Cantidad de Graficas Filas y Columnas    
ax.title.set_text('First Plot')
ax.set_xlabel('common xlabel')
ax.set_ylabel('common ylabel')
ax.scatter(X,Y)
ax.plot(X, Y_PREDICT, color='red');
plt.show()

#Preddicion
Y_NEW_PREDICTION = model.predict([[2030]])
print(Y_NEW_PREDICTION)


########################## Metodo guardando Imagen
#plt.xlabel(columnNameX)
#plt.ylabel(columnNameY)
#plt.title("Ploteo de Puntos")
#plt.scatter(X, Y)
#plt.savefig(PATH_FINAL_REPORT)
#plt.close()

#image = Image.open(PATH_FINAL_REPORT)
#st.subheader("Resultados")
#st.image(image, caption='Correlacion de Puntos, Sobre las Columnas Seleccioandas')