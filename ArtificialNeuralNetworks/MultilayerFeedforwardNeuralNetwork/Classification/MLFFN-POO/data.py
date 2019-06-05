# Importando librerías
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from util import Util

class Data(object):

    def preProcessData(self, train_file, final_test_size, labelEncodeX1name, labelEncodeX2name, scalerName):
        # Importando Datasets
        dataset_csv = pd.read_csv(train_file)

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

        util = Util()
        # Guardando LabelEncoder a disco
        util.saveFile(labelEncoder_X_1,labelEncodeX1name)
        util.saveFile(labelEncoder_X_2,labelEncodeX2name)

        # Escalamiento/Normalización de Features (StandardScaler: (x-u)/s)
        stdScaler = StandardScaler()
        dataset_values[:,0:10] = stdScaler.fit_transform(dataset_values[:,0:10])

        # Guardando StandardScaler a disco
        util.saveFile(stdScaler,scalerName)

        # Dataset final normalizado
        dataset_final = pd.DataFrame(dataset_values,columns=dataset_columns, dtype=np.float64)

        print ("\nDataset Final:")
        print(dataset_final.describe(include='all'))
        print("\n", dataset_final.head())

        # Distribuciones de la data y Correlaciones
        util.plotHistogram(dataset_final)
        util.plotCorrelations(dataset_final)

        # Obteniendo valores a procesar
        X = dataset_final.iloc[:, 0:10].values
        y = dataset_final.iloc[:, 10].values

        # Dividiendo el Dataset en sets de Training y Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = final_test_size, random_state = 0)

        return X_train, X_test, y_train, y_test
