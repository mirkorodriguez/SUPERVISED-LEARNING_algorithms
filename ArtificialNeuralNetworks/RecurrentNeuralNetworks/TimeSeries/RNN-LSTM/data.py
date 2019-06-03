import numpy as np
from util import Util

class Data(object):    

    def preProcessData(self, train_file, test_file, serie_name_value, amplitude):
        
        # Serie a predecir
        SERIE_NAME_VALUE = serie_name_value
    
        # Feature Scaling
        [training_set_scaled, test_set_scaled, sc] = Util().scaleData(train_file, test_file, SERIE_NAME_VALUE)
        
        # Estructura de 60 datos de entrada (t) y 1 dato de salida (t+1)  X[1-60] --> y[61]
        AMPLITUDE = amplitude
        TRAIN_LENGTH = training_set_scaled.shape[0];
        
        X_train = []
        y_train = []
        
        for i in range(AMPLITUDE, TRAIN_LENGTH):
            X_train.append(training_set_scaled[i-AMPLITUDE:i, 0])
            y_train.append(training_set_scaled[i, 0])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # Redimensionando X_train (1198,60,1)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
        return X_train, y_train  

