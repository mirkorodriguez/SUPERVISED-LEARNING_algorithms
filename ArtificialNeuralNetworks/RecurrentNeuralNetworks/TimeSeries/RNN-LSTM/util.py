import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Util(object):

    def __init__(self):
        print(' ')

    def scaleData(self, train_file, test_file, serie_name_value):

        dataset_train = pd.read_csv(train_file)
        dataset_test = pd.read_csv(test_file)

        SERIE_NAME_VALUE = serie_name_value

        # Estandarizacion: (x-u)/dev
        # Normalizacion: (x-min)/(max-min) --> feature_range [0 - 1]
        sc = MinMaxScaler(feature_range = (0, 1))
        total_data = pd.concat((dataset_train[SERIE_NAME_VALUE], dataset_test[SERIE_NAME_VALUE]), axis = 0)
        total_data = total_data.values.reshape([-1,1])
        sc.fit(total_data)

        training_set = dataset_train.loc[:, [SERIE_NAME_VALUE]].values
        training_set_scaled = sc.transform(training_set)
        
        test_set = dataset_test.loc[:, [SERIE_NAME_VALUE]].values
        test_set_scaled = sc.transform(test_set)

        return training_set_scaled, test_set_scaled, sc


    def graphResults(self, real_stock_price,predicted_stock_price,serie_name_label,serie_name_value):
    
        SERIE_NAME_LABEL = serie_name_label
        SERIE_NAME_VALUE = serie_name_value
    
        # Visualizando resultados
        plt.figure(figsize=(14,10),dpi=80)
        plt.plot(real_stock_price, color = 'blue', label = SERIE_NAME_LABEL + ' | ' + SERIE_NAME_VALUE + ' - (Real)')
        plt.plot(predicted_stock_price, color = 'red', label = SERIE_NAME_LABEL + ' | ' + SERIE_NAME_VALUE + ' - (Predicted)', dashes=[6, 2])
        plt.title('Prediction:' + SERIE_NAME_LABEL + ' | ' + SERIE_NAME_VALUE)
        plt.xlabel('Tiempo')
        plt.ylabel(SERIE_NAME_LABEL)
        plt.legend()
        plt.show()
