import numpy as np

import os

import re

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import tensorflow as tf 

import keras

from tensorflow.keras.utils import to_categorical

from keras.models import Sequential,Input,Model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

dirname = os.path.join(os.getcwd(), 'plant_leaves')

img_path = dirname + os.sep
images = []
directories = []
dir_count = []
prev_root = ''
cant = 0

for root, dirs, filenames in os.walk(".", topdown=False):
   for filename in filenames:
      #print(os.path.join(root, name))
     if re.search("\.(JPG|jpg|jpeg|png|bmp|tiff)$", filename):
         cant = cant + 1
         print(filename)
         file_path = os.path.join(root, filename)
         image = plt.imread(file_path)
         images.append(image)
         b = "Leyendo... " + str(cant)
         if prev_root != root:
             prev_root = root
             directories.append(root)
             dir_count.append(cant)
             cant = 0
dir_count.append(cant)
dir_count = dir_count[1:]
dir_count[0] = dir_count[0]+1
print("Directorios leídos: ", len(directories))
print("Imágenes de cada direcotrio: ", dir_count)
print("Total de imágenes en subdirectorio: ", sum(dir_count))

labels = []
indice = 0
for cantidad in dir_count:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1
print("Cantidad etiquetas creadas: ", len(labels))

ninhos = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice, name[len(name)-1])
    ninhos.append(name[len(name)-1])
    
y = np.array(labels)
x = np.array(images, dtype=np.uint8)

classes = np.unique(y)
n_classes = len(classes)

print("Total de outputs: ", n_classes)
print("Output classes: ", classes)

train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2)
print("Datos de aprendizaje: ", train_X.shape,train_Y.shape)
print("Testing de aprendizaje: ", test_X.shape,test_Y.shape)

plt.figure(figsize=[5,5])

plt.subplot(122)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth: {}".format(test_Y[0]))

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.0
test_X = test_X / 255.0

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print('Etiqueta original:', train_Y[0])

print('Después de la conversión a one-hot:', train_Y_one_hot[0])

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)


INIT_LR = 1e-3
epochs = 6
batch_size = 64

ninhos_model = Sequential()
ninhos_model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', input_shape=(4000, 6000, 3)))
ninhos_model.add(LeakyReLU(alpha=0.1))
ninhos_model.add(MaxPooling2D((2,2),padding='same'))
ninhos_model.add(Dropout(0.5))

ninhos_model.add(Flatten())
ninhos_model.add(Dense(4, activation='relu'))
ninhos_model.add(LeakyReLU(alpha=0.1))
ninhos_model.add(Dropout(0.5))
ninhos_model.add(Dense(n_classes, activation='softmax'))

ninhos_model.build()
ninhos_model.summary()
ninhos_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adagrad(learning_rate=INIT_LR, decay=INIT_LR / 100), metrics=['accuracy'])



ninhos_train = ninhos_model.fit(train_X, train_label, batch_size=10, epochs=6, verbose=1, validation_data=(valid_X, valid_label))
ninhos_model.save("ninhos_mnist.h5py")

test_eval = ninhos_model.evaluate(test_X, test_Y_one_hot, verbose=1)

 

print('TEST:')

print('Perdida:', test_eval[0])

print('Exactitud:', test_eval[1])