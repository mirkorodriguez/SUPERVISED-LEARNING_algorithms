#https://www.kaggle.com/benhamner/popular-datasets-over-time

# --------------------
# Importando librerias
# --------------------

# Keras model module
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
# MNIST data
from keras.datasets import mnist
# Plotter
from matplotlib import pyplot

# ----------------------
# Cargando dataset MNIST
# ----------------------

# input image dimensions
img_rows, img_cols = 28, 28

# /Applications/Python 3.6/Install Certificates.command
# conda update openssl
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plotting first sample of X_train
pyplot.imshow(X_train[0], cmap='gray')

print(X_train.shape)
print(X_test.shape)

# -------------------------------------------
# Preprocesando la data de entrada para Keras
# -------------------------------------------
# Reshape input data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

print(X_train.shape)
print(X_test.shape)

print ("Data sin normalizar:\n", X_train[0][14])

# Normalizando la data --> decimal y el el rango de 0-1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print ("Data Normalizada:\n", X_train[0][14])

# ----------------------------------------
# Categorizando las salidas: Clases de 0-9
# ----------------------------------------

# Preprocess class labels
# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train_cat = np_utils.to_categorical(y_train, 10)
y_test_cat = np_utils.to_categorical(y_test, 10)

print (y_train[0], " categorizado es:\n ", y_train_cat[0])

# --------------------------------------
# Definiendo la Arquitectura del modelo
# --------------------------------------

#Declare Sequential model
model = Sequential()


# CNN input layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(1,1), input_shape = (img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Adding more CNN layers
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="valid", strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Adding more CNN layers
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="valid", strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Fully connected Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# --------------------------------
# Compilando y entenando el modelo
# --------------------------------

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit Keras model
model.fit(X_train, y_train_cat, batch_size=128, epochs=5, verbose=1)

# -------------------
# Evaluando el modelo
# -------------------

# Evaluate Keras model
print('\nEvaluando el modelo:')

score = model.evaluate(X_test, y_test_cat, verbose=1)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])


# ----------------------------------------------------
# Ejercicio: Predecir una imagen creada por el usuario
# ----------------------------------------------------
from keras.preprocessing import image
test_image_path = './sample/numberTest2.jpg'
test_image_original = image.load_img(test_image_path)
pyplot.imshow(test_image_original)
pyplot.show()

test_image = image.load_img(test_image_path,target_size = (28, 28), grayscale = True)
test_image = image.img_to_array(test_image)
test_image = test_image.reshape(1,28,28,1)
test_image = test_image.astype('float32')
test_image /= 255

predicions = model.predict(test_image)
print("\nPredictions:\n", predicions)

predicion_class = model.predict_classes(test_image)
print("\nPredictions Class: ", predicion_class)
