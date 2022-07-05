from base64 import encode
from pickle import TRUE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.tree  import plot_tree

#LabelEncoder
le = preprocessing.LabelEncoder()





# Importacion de Datos
outlook =   ['sunny','sunny','overcast','rain','rain','rain','overcast',
            'sunny','sunny','rain','sunny','overcast','overcast','rain']
temperature=['hot','hot','hot','mild','cool','cool','cool',
             'mild','cool','mild','mild','mild','hot','mild']
humidity =  ['high','high','high','high','normal','normal','normal',
            'high','normal','normal','normal','high','normal','high']
windy =     ['false','true','false','false','false','true','true',
            'false','false','false','true','true','false','true']
play =      ['N','N','P','P','P','N','P','N','P','P','P','P','P','N']

outlook_encoded = le.fit_transform(outlook)
temperature_encoded = le.fit_transform(temperature)
humidity_encoded = le.fit_transform(humidity)
windy_encoded = le.fit_transform(windy)
play_encoded = le.fit_transform(play)

# Combinacion de atributos en una lista de tuplas
print(outlook_encoded)
features = list(zip(outlook_encoded, temperature_encoded, humidity_encoded, windy_encoded))
print(features)

# Creamos clasificador gaussiano
model = GaussianNB()

# Entrenamos el modelo
model.fit(features, play_encoded)

predict = model.predict([[2,1,0,0]])

predict_class = le.inverse_transform(predict)

print("Predict Value: {0}".format(predict_class[0]))
