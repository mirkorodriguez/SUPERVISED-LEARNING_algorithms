
#-------------------------------------
# PARTE I - Preprocesamiento de la Data
#-------------------------------------

# Importando librerías
import pandas as pd
import numpy as np
from util import plot_confusion_matrix, saveFile, plotHistogram, plotCorrelations, loadFile
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importando Datasets
dataset_csv = pd.read_csv('../dataset/clientes_data.csv')

# Describir la data original
print ("\nDataset original:\n", dataset_csv.describe(include='all'))

# Dataset reducido
dataset = dataset_csv.iloc[:,3:14]
dataset_columns = dataset.columns
dataset_values = dataset.values

# Describir la data truncada
print ("\nDataset reducido: \n", dataset.describe(include='all'))
print("\n",dataset.head())

# Codificando datos categóricos
labelEncoder_X_1 = LabelEncoder()
dataset_values[:, 1] = labelEncoder_X_1.fit_transform(dataset_values[:, 1])
labelEncoder_X_2 = LabelEncoder()
dataset_values[:, 2] = labelEncoder_X_2.fit_transform(dataset_values[:, 2])
print ("\nDataset Categorizado: \n", dataset_values)

# Guardando LabelEncoder a disco
saveFile(labelEncoder_X_1,"./model/labelEncoder_X_1.save")
saveFile(labelEncoder_X_2,"./model/labelEncoder_X_2.save")

# Escalamiento/Normalización de Features (StandardScaler: (x-u)/s)
stdScaler = StandardScaler()
dataset_values[:,0:10] = stdScaler.fit_transform(dataset_values[:,0:10])
# print("\nStandardScaler: ", stdScaler)
# print("No Observations: ", stdScaler.n_samples_seen_)
# print("Mean: ", stdScaler.mean_)
# print("Varianza: ", stdScaler.var_)
# print ("\nDataset Normalizado\n: ", dataset_values)

# Guardando StandardScaler a disco
saveFile(stdScaler,"./model/stdScaler.save")

# Dataset final normalizado
dataset_final = pd.DataFrame(dataset_values,columns=dataset_columns, dtype=np.float64)

print ("\nDataset Final:")
print(dataset_final.describe(include='all'))
print("\n", dataset_final.head())

# Distribuciones de la data y Correlaciones
plotHistogram(dataset_final)
plotCorrelations(dataset_final)

# Obteniendo valores a procesar
X = dataset_final.iloc[:, 0:10].values
y = dataset_final.iloc[:, 10].values

# Dividiendo el Dataset en sets de Training y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#----------------------------------------------------
# PARTE II - Construyendo la Red Neuronal Artificial!
#----------------------------------------------------

# Importando Keras y Tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform

# Inicializando la Red Neuronal
neural_network = Sequential()

# kernel_initializer Define la forma como se asignará los Pesos iniciales Wi
initial_weights = RandomUniform(minval = -0.5, maxval = 0.5)

# Agregado la Capa de entrada y la primera capa oculta
# 10 Neuronas en la capa de entrada y 8 Neuronas en la primera capa oculta
neural_network.add(Dense(units = 8, kernel_initializer = initial_weights, activation = 'relu', input_dim = 10))

# Agregando capa oculta
neural_network.add(Dense(units = 5, kernel_initializer = initial_weights, activation = 'relu'))

# Agregando capa oculta
neural_network.add(Dense(units = 4, kernel_initializer = initial_weights, activation = 'relu'))

# Agregando capa de salida
neural_network.add(Dense(units = 1, kernel_initializer = initial_weights, activation = 'sigmoid'))

# Imprimir Arquitectura de la Red
neural_network.summary()

# Compilando la Red Neuronal
# optimizer: Algoritmo de optimización | binary_crossentropy = 2 Classes
# loss: error
neural_network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Entrenamiento
neural_network.fit(X_train, y_train, batch_size = 32, epochs = 100)




#-----------------------------------------------
# PARTE III - Predicciones y evaluando el modelo
#-----------------------------------------------

# standarScaler = loadFile("./model/stdScaler.save")
# print("StandardScaler: ",standarScaler)
# print("No Observations: ", standarScaler.n_samples_seen_)
# print("Mean: ", standarScaler.mean_)
# print("Varianza: ", standarScaler.var_)

# Haciendo predicción de los resultados del Test
y_pred = neural_network.predict(X_test)
y_pred_norm = (y_pred > 0.5)

y_pred_norm = y_pred_norm.astype(int)
y_test = y_test.astype(int)

# 50 primeros resultados a comparar
print("\nPredicciones (50 primeros):")
print("\n\tReal", "\t", "Predicción(N)","\t", "Predicción(O)")
for i in range(50):
    print(i, '\t', y_test[i], '\t ', y_pred_norm[i], '\t \t', y_pred[i])

# Aplicando la Matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_norm)
print ("\nMatriz de Confusión: \n", cm)

# Graficando la Matriz de Confusión
plot_confusion_matrix(y_test, y_pred_norm, normalize=False,title="Matriz de Confusión: Abandono de clientes")
