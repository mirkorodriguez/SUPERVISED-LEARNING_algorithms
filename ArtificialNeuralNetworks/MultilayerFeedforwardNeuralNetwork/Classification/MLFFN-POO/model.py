# Importando Keras y Tensorflow
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.initializers import RandomUniform

import datetime

from util import Util

class Model(object):

    # Atributos
    neural_network = None

    # Metodos
    def __init__(self):
        # inicializando the RNN
        self.neural_network = Sequential()

    def setModel(self, new_ann_model):
        self.neural_network = new_ann_model

    def trainNetwork(self, final_X_train,final_y_train, final_epochs, final_batch_size):
        print("\n")
        print('Iniciando ENTRENAMIENTO a las: ', datetime.datetime.now())
        print("...")

        # kernel_initializer Define la forma como se asignará los Pesos iniciales Wi
        initial_weights = RandomUniform(minval = -0.4, maxval = 0.4)

        # Agregado la Capa de entrada y la primera capa oculta
        # 10 Neuronas en la capa de entrada y 8 Neuronas en la primera capa oculta
        self.neural_network.add(Dense(units = 8, kernel_initializer = initial_weights, activation = 'relu', input_dim = 10))
        # self.neural_network.add(Dropout(p = 0.1))

        # Agregando capa oculta
        self.neural_network.add(Dense(units = 6, kernel_initializer = initial_weights, activation = 'relu'))
        # self.neural_network.add(Dropout(p = 0.1))

        # Agregando capa oculta
        self.neural_network.add(Dense(units = 4, kernel_initializer = initial_weights, activation = 'relu'))

        # Agregando capa de salida
        self.neural_network.add(Dense(units = 1, kernel_initializer = initial_weights, activation = 'sigmoid'))

        # Imprimir Arquitectura de la Red
        self.neural_network.summary()

        # Compilando la Red Neuronal
        # optimizer: Algoritmo de optimización | binary_crossentropy = 2 Classes
        # loss: error
        self.neural_network.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Entrenamiento
        self.neural_network.fit(final_X_train,final_y_train, batch_size = final_batch_size, epochs = final_epochs)

        print("...")
        print('Terminando ENTRENAMIENTO a las: ', datetime.datetime.now())


    def saveModel(self, model_name_h5):
        # Guardar el modelo en disco: Model and Architecture to single file
        final_model_name = "./model/" + model_name_h5
        self.neural_network.save(final_model_name)
        return final_model_name


    def loadModel(self, model_file_name):
        print("\nModelo a cargar:", model_file_name)
        loaded_model = load_model( "./model/" + model_file_name)
        return loaded_model


    def validateModel(self, X_test, y_test):

        # Haciendo predicción de los resultados del Test
        y_pred = self.neural_network.predict(X_test)
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
        return cm, y_pred_norm
