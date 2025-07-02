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
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D, SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img, array_to_img

(x_train,y_train), (x_test,y_test) = cifar10.load_data()


print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

plt.figure(figsize=(10,5))
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.imshow(x_train[i])
  plt.title(f"index {i}, label {y_train[i][0]}")
  plt.axis("off")
plt.show()

#normalization 0 ile 1 arasında scale etme
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#one hot encoding
y_train = to_categorical(y_train, 10) #10 = sınıf sayısı
y_test = to_categorical(y_test, 10)

#ann modelinin oluşturulması ve derlenmesi
model = Sequential()

model.add(Flatten(input_shape=(32, 32, 3))) # 3D den 1D ye dönüştürme

model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#callback fonksiyonlarının tanımlanması ve modelin eğitilmesi
#early stopping = erken durdurma
#monitor = doğrulama(validasyon) setindeki kaybı(loss) izler.
#patience = epoch boyunca val loss değişmiyorsa erken durdurma yapar.
#restore_best_weights = en iyi modelin ağırlıklarını geri yükler.
early_stopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

#model checkpoint = en iyi modelin ağırlıklarını kaydeder.
model_checkpoint = ModelCheckpoint('best_model.h5',monitor='val_loss',save_best_only=True)

#model training
history = model.fit(x_train, y_train, #train verisetini model eğitmek için verdik
                    epochs=20, #model toplamda 20 sefer verisetini görecek
                    batch_size=64, #veriler 64 lük paketler halinde işlenecek
                    validation_split = 0.2, #eğitim verisinin %20 si doğrulama olarak kullanılacak
                    callbacks=[early_stopping, model_checkpoint])

#test dataseti ile model performans değerlendirmesi
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

#training ve validation accuracy görselleştirmesi
plt.figure()
plt.plot(history.history['accuracy'],marker = "o", label='Training Accuracy')
plt.plot(history.history['val_accuracy'],marker = "o", label='Validation Accuracy')
plt.title("ANN accuracy on CIFAR-10 Dataset")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#trainin ve validation loss görselleştirme
plt.figure()
plt.plot(history.history['loss'],marker = "o", label='Training loss')
plt.plot(history.history['val_loss'],marker = "o", label='Validation loss')
plt.title("ANN loss on CIFAR-10 Dataset")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#modeli kaydetme
model.save("final_cifar10_ann_model.keras")

loaded_model = load_model("/C:/Users/cirim/.spyder-py3/final_cifar10_ann_model.keras")

#test dataseti ile model performans değerlendirmesi
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
