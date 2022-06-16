import tensorflow as tf
import os
import matplotlib.pyplot as plt
import re
import numpy as np
from PIL import Image

dirname = os.path.join(os.getcwd(), 'plants_leaves')

img_path = dirname + os.sep
images = []
categorias = []
labels = []

categorias = os.listdir(dirname)
x=0
for direcotrio in categorias:
    print(direcotrio)
    for imagen in os.listdir(dirname + '/' + direcotrio):
        img = Image.open(dirname + '/' + direcotrio + '/' + imagen).resize((200,200))
        img = np.array(img)
        images.append(img)
        labels.append(x)
    x += 1

images = np.asanyarray(images)
images.shape
images = images[:,:,:,0]
images.shape

plt.figure()
plt.imshow(images[9])
plt.colorbar()
plt.grid(False)
plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(200, 200)),    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
labels = np.asarray(labels)
model.fit(images, labels, epochs=100)

im=0
im = Image.open('C:\\Users\\Jackson\\Documents\\IA - Reconocimiento de imágenes\\plants_leaves\\healthy\\0006_0008.JPG').resize((200,200))
im = np.asarray(im)
im = im[:,:,0]
im = np.array([im])
im.shape
test = im

predicciones = model.predict(test)

print(predicciones)

predict = categorias[np.argmax(predicciones[0])]

test_loss, test_acc = model.evaluate(images,  labels, verbose=2)

print('\nTest accuracy:', test_acc)


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(categorias[predicted_label],
                                100*np.max(predictions_array),
                                categorias[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predicciones[i], labels, im)
plt.subplot(1,2,2)
plot_value_array(i, predicciones[i],  labels)
plt.show()
