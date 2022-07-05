
import streamlit as st
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

import Helpers.HandlerFiles as Files
import Helpers.HandlerParameters as Parameters


def information():
    st.sidebar.write("## Redes Neuronales 📑")

    st.sidebar.write("¿Qué es?")
    st.sidebar.write("Una red neuronal es un método de la inteligencia artificial que enseña a las computadoras a procesar datos de una manera que está  " +
                "inspirada en la forma en que lo hace el cerebro humano. Se trata de un tipo de proceso de machine learning llamado aprendizaje profundo, " +
                "que utiliza los nodos o las neuronas interconectados en una estructura de capas que se parece al cerebro humano. Crea un sistema adaptable " +
                "que las computadoras utilizan para aprender de sus errores y mejorar continuamente. De esta forma, las redes neuronales artificiales intentan  " +
                "resolver problemas complicados, como la realización de resúmenes de documentos o el reconocimiento de rostros, con mayor precisión.")
    
    st.sidebar.write("Las redes neuronales pueden ayudar a las computadoras a tomar decisiones inteligentes con asistencia humana limitada. Esto se debe a que " +
                "pueden aprender y modelar las relaciones entre los datos de entrada y salida que no son lineales y que son complejos. Por ejemplo, " +
                "pueden realizar las siguientes tareas.")


def algoritmo():
    information()

    st.title("Algoritmo - Redes Neuronales")

    st.subheader("Carga de Archivo")
    df = Files.uploadFile("Seleccione el Archivo")

    if df is not None:
        st.write("Información Cargada")
        st.dataframe(df)

        # Pedimos los parametros
        st.markdown("""---""")
        st.subheader("Parámetros")
        columnsClassifiquer = Parameters.SelectColumns("Seleccione Las Columnas Para Clasificar", df)
        columnClass = Parameters.SelectAxis("Seleccione Columna Clase", df)
        solver = Parameters.SolverNeuralNetwork("Seleccione el Tipo de Solucionador a Utilizar")

        if len(columnsClassifiquer) > 0:
            try:
                st.write("Seleccione Valores para Predición, Según las Columnas Seleccionadas")
                cols = st.columns(len(columnsClassifiquer))
                predictValues = []
                count = 0
                for i in columnsClassifiquer:
                    col = cols[count]
                    predictValues.append(col.selectbox(str(i), df[i].unique(), 0))
                    count = count + 1

                resButton = st.button("Ejecutar Algoritmo")
                if resButton :
                    # Comenzammos codificando las clases
                    le = preprocessing.LabelEncoder()
    
                    # Codificamos valores para las columnas seleccionadas y los valores de prediccion
                    encodedList = []
                    encodedPredict = []
                    count = 0
                    for i in columnsClassifiquer:
                        encodedList.append(le.fit_transform(df[i]))
                        encodedPredict.append(np.where(le.classes_ == predictValues[count])[0][0])  # buscamos clase para prediccion
                        count = count + 1
    
                    #Hacemos tuplas con la codificacion anterior
                    features = list(zip(*encodedList))
    
                    # Codificamos columna Clase
                    encodedClass = le.fit_transform(df[columnClass])
    
                    # Creacion y entrenamiendo del modelo
                    mlp = MLPClassifier(solver=solver, max_iter=500, hidden_layer_sizes=(100,100,100), random_state=0, verbose=10).fit(features, encodedClass)
                    prediction = mlp.predict([encodedPredict])
                    predictClass = le.inverse_transform(prediction)
                    
                    #Prediccion 
                    prediction = mlp.predict([encodedPredict])
                    predictClass = le.inverse_transform(prediction)
    
                    #Resultados
                    st.markdown("""---""")
                    st.subheader("Resultados")
                    st.write("Matriz Codificada para Columnas de Clasificación Seleccionadas")
                    st.dataframe(features)
    
                    st.write("Valores de Predicción Codificados")
                    predictStr = ""
                    countTemp = 0
                    for i in predictValues:
                        predictStr += "{0} \\rightarrow {1} \\newline \n".format(i, encodedPredict[countTemp])
                        countTemp = countTemp + 1
                    
                    st.latex(predictStr)
    
                    st.write("Prediccion")
                    predictionStr= "Predicción \\rightarrow {0}".format(predictClass[0])
                    st.latex(predictionStr)

            except Exception as e:
                st.warning("No Se A Podido Ejecutar La Operación, Seleccione Nuevos Parametros y Vuelva a Intentar")
                print(e)


algoritmo()
