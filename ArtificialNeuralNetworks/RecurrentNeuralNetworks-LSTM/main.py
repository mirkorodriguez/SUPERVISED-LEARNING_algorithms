
#====================================
# REDES NEURONALES RECURRENTES - LSTM
#====================================
from ann_model import Model
from data import Data
from util import Util

#-----------
# PARAMETROS
#-----------
TRAIN_DATASET_FILE = 'dataset/BVL-SIDERC1/Sider-Peru_Stock-SIDERC1_Train.csv'
TEST_DATASET_FILE = 'dataset/BVL-SIDERC1/Sider-Peru_Stock-SIDERC1_Test.csv'
SERIE_NAME_VALUE = 'Apertura'
SERIE_NAME_LABEL = 'EMPRESA SIDERURGICA DEL PERU S.A.A.'

EPOCHS = 250
BATCH_SIZE = 48
AMPLITUDE = 60
NAME_MODEL_TO_SAVE = 'rnn-lstm_model-sider25'

#------------------------------------
# PARTE I - PREPROCESAMIENTO DE DATOS
#------------------------------------
[X_train, y_train] = Data().preProcessData(TRAIN_DATASET_FILE,
                                        TEST_DATASET_FILE,
                                        SERIE_NAME_VALUE,
                                        AMPLITUDE)
print("X_train: ", X_train)
print("y_train: ", y_train)

#---------------------------------------------------
# PARTE II - CONSTRUYENDO LA RED NEURONAL RECURRENTE
#---------------------------------------------------
model = Model()
model.trainNetwork(X_train, y_train, EPOCHS, BATCH_SIZE)
# Summarize model.
model.rnn_model.summary()

#-----------------------------------------------
# PARTE III - GUARDAR LA RED NEURONAL RECURRENTE
#-----------------------------------------------
# Saving Model to disk
model_file_name = model.saveModel(NAME_MODEL_TO_SAVE)
print("Modelo guardado en disco >> ", model_file_name)

#-----------------------------------------------
# PARTE IV - CARGAR LA RED NEURONAL RECURRENTE
#-----------------------------------------------
from ann_model import Model
new_model = Model()
FILENAME_MODEL_TO_LOAD = model_file_name
loaded_model = new_model.loadModel(FILENAME_MODEL_TO_LOAD)
print("Modelo cargado de disco << ", loaded_model)
# Summarize model.
loaded_model.summary()

new_model.setModel(loaded_model)

#-----------------------------------------------------
# PARTE V - PREDICCIONES Y VISUALIZACION DE RESULTADOS
#-----------------------------------------------------
[rmse,real_stock_serie,predicted_stock_serie] = new_model.validateModel(TRAIN_DATASET_FILE,
                                                                    TEST_DATASET_FILE,
                                                                    SERIE_NAME_VALUE,
                                                                    AMPLITUDE)
print("rmse: ", rmse)

Util().graphResults(real_stock_serie,
                  predicted_stock_serie,
                  SERIE_NAME_LABEL,
                  SERIE_NAME_VALUE)
