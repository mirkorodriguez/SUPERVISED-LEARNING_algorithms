# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from sklearn.metrics import mean_squared_error

import math
import datetime
import numpy as np
import pandas as pd

from util import Util

class Model(object):

    # Atributos
    rnn_model = None

    # Metodos
    def __init__(self):
        # inicializando the RNN
        self.rnn_model = Sequential()

    def setModel(self, new_rnn_model):
        self.rnn_model = new_rnn_model
        
    def trainNetwork(self, final_X_train,final_y_train, final_epochs, final_batch_size):
        print("\n")
        print('Iniciando ENTRENAMIENTO a las: ', datetime.datetime.now())
        print("...")

        # 1ra capa LSTM y Dropout para regularizacion.
        # input_shape (60,1)
        self.rnn_model.add(LSTM(units = 50, return_sequences = True, input_shape = (final_X_train.shape[1], 1)))
        # 20% de las neuronas seran ignoradas durante el training (20%xunits = 10)
        # Para hacer menos probable el overfiting
        self.rnn_model.add(Dropout(0.2))

        # 2da capa LSTM y Dropout para regularizacion.
        self.rnn_model.add(LSTM(units = 50, return_sequences = True))
        self.rnn_model.add(Dropout(0.2))

        # 3ra capa LSTM y Dropout para regularizacion.
        self.rnn_model.add(LSTM(units = 50, return_sequences = True))
        self.rnn_model.add(Dropout(0.2))

        # 4ta capa LSTM y Dropout para regularizacion.
        self.rnn_model.add(LSTM(units = 50, return_sequences = False))
        self.rnn_model.add(Dropout(0.2))

        # Output layer
        self.rnn_model.add(Dense(units = 1))

        # Compiling the RNN
        self.rnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Entrenamiento de la RNN con nuestro Training set
        self.rnn_model.fit(final_X_train, final_y_train, epochs = final_epochs, batch_size = final_batch_size)

        print("...")
        print('Terminando ENTRENAMIENTO a las: ', datetime.datetime.now())


    def saveModel(self, model_name_h5):
        # Guardar el modelo en disco: Model and Architecture to single file
        final_model_name = "./model/" + model_name_h5 + ".h5"
        self.rnn_model.save(final_model_name)
        return final_model_name


    def loadModel(self, model_file_name):
        print("\nModelo a cargar:", model_file_name)
        loaded_model = load_model(model_file_name)
        return loaded_model


    def validateModel(self, train_file, test_file, serie_name_value, amplitude):

        # Serie a predecir
        SERIE_NAME_VALUE = serie_name_value
        AMPLITUDE = amplitude

        # Precio de Accion reales
        dataset_train = pd.read_csv(train_file)
        dataset_test = pd.read_csv(test_file)
        real_stock_price = dataset_test.loc[:, [SERIE_NAME_VALUE]].values

        # Feature Scaling
        [training_set_scaled, test_set_scaled, sc] = Util().scaleData(train_file, test_file, SERIE_NAME_VALUE)

        # Concatenar en un solo vector (train + test)
        dataset_total = np.concatenate((training_set_scaled, test_set_scaled), axis = 0)
        inputs = dataset_total[len(training_set_scaled) - AMPLITUDE:] 
        # Redimensionando de (80,) --> (80,1)
        inputs = inputs.reshape(-1,1)

        TEST_LENGTH = dataset_test.shape[0];

        X_test = []

        for i in range(AMPLITUDE, AMPLITUDE + TEST_LENGTH):
            X_test.append(inputs[i-AMPLITUDE:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_stock_price = self.rnn_model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        # RMSE
        rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        return rmse,real_stock_price,predicted_stock_price
