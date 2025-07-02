# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:32:31 2025

@author: cirim
"""
"""
Müsait GPU ların ekrana bastırılması ve GPU yu aktif etme işlemi.

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')  # İlk GPU'yu kullanır


"""
import cv2
import urllib
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import random , os , glob
from imutils import paths
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D, SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img, array_to_img
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Veri setini yükle
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Görüntüleri CNN için uygun formata getir (28x28x1)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalizasyon (0-255 arası değerleri 0-1 arasına ölçekle)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Etiketleri one-hot encoding formatına dönüştür
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()

# 1. Evrişim Katmanı
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# 2. Evrişim Katmanı
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Tam Bağlı Katmanlar
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 sınıf (0-9 rakamlar)

# Modeli derle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model özeti
model.summary()

history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Doğruluğu: {test_acc:.4f}")

# Eğitim ve doğrulama kayıpları
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğrulukları
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

predictions = model.predict(test_images)

# İlk test görüntüsünün tahminini göster
import numpy as np
print(f"Tahmin Edilen Sınıf: {np.argmax(predictions[0])}")
print(f"Gerçek Sınıf: {np.argmax(test_labels[0])}")

# İlk test görüntüsünü görselleştir
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Tahmin: {np.argmax(predictions[0])}, Gerçek: {np.argmax(test_labels[0])}")
plt.show()

#modeli kaydetme
model.save("final_mnist_ann_model.keras")