
#=======================================
# REDES NEURONALES ARTIFICIALES - MLFFNN
#=======================================
from model import Model
from data import Data
from util import Util

#-----------
# PARAMETROS
#-----------
DATASET_FILE = '../dataset/clientes_data.csv'

EPOCHS = 500
BATCH_SIZE = 25
TEST_SIZE = 0.2 # 20% Test Set

NAME_MODEL_TO_SAVE = 'ann_model-250-50.h5'
NAME_LABELENCODER_X1_TO_SAVE = 'labelEncoder_X_1.save'
NAME_LABELENCODER_X2_TO_SAVE = 'labelEncoder_X_2.save'
NAME_SCALER_TO_SAVE = 'stdScaler.save'

#------------------------------------
# PARTE I - PREPROCESAMIENTO DE DATOS
#------------------------------------
[X_train, X_test, y_train, y_test] = Data().preProcessData(DATASET_FILE,
                                        TEST_SIZE,
                                        NAME_LABELENCODER_X1_TO_SAVE,
                                        NAME_LABELENCODER_X2_TO_SAVE,
                                        NAME_SCALER_TO_SAVE)
print("X_train: ", X_train)
print("y_train: ", y_train)
print("X_test: ", X_test)
print("y_test: ", y_test)

#---------------------------------------------------
# PARTE II - CONSTRUYENDO LA RED NEURONAL ARTIFICIAL
#---------------------------------------------------
model = Model()
model.trainNetwork(X_train, y_train, EPOCHS, BATCH_SIZE)
# Summarize model.
model.neural_network.summary()
model.graphModel("./model/model_graph.png")
# Guardar la RNA en disco
model_file_name = model.saveModel(NAME_MODEL_TO_SAVE)
print("Modelo guardado en disco >> ", model_file_name)


#-----------------------------------------------------
# PARTE III - PREDICCIONES Y VISUALIZACION DE RESULTADOS
#-----------------------------------------------------

FILENAME_MODEL_TO_LOAD = NAME_MODEL_TO_SAVE
FILENAME_SCALER_TO_LOAD = NAME_SCALER_TO_SAVE
FILENAME_LABELENCODER_X1_TO_LOAD = NAME_LABELENCODER_X1_TO_SAVE
FILENAME_LABELENCODER_X2_TO_LOAD = NAME_LABELENCODER_X2_TO_SAVE

# Cargar la RNA desde disco
from model import Model
new_model = Model()
loaded_model = new_model.loadModel(FILENAME_MODEL_TO_LOAD)
print("Modelo cargado de disco << ", loaded_model)
# Summarize model.
loaded_model.summary()

new_model.setModel(loaded_model)

new_util = Util()
# Cargar los parametros usados
loaded_scaler = new_util.loadFile(FILENAME_SCALER_TO_LOAD)
loaded_labelEncoderX1 = new_util.loadFile(FILENAME_LABELENCODER_X1_TO_LOAD)
loaded_labelEncoderX2 = new_util.loadFile(FILENAME_LABELENCODER_X2_TO_LOAD)

# Validar y predecir
[cm, y_pred] = new_model.validateModel(X_test, y_test)
print ("\nMatriz de Confusión: \n", cm)

# Graficando la Matriz de Confusión
title = "Matriz de Confusión: Abandono de clientes"
new_util.plot_confusion_matrix(y_test, y_pred, normalize=False,title=title)
