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
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img, array_to_img
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import layers, models


#verisetini yükle
max_features = 10000
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words = max_features)

#veriyi padding ile aynı uzunluğa sabitleyelim
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


#model oluşturma
def build_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 64, input_length = maxlen))
    model.add(LSTM(units = 8))
    model.add(Dropout(0.6))
    model.add(Dense(1,activation="sigmoid")) #2 sınıf olduğu için kullandık
    
    #derleme
    model.compile(optimizer = Adam(learning_rate=0.0001),
                  loss = "binary_crossentropy",
                  metrics = ["accuracy"])
    return model

model = build_lstm_model()
model.summary()

#earlystopping tanımlama 
early_stopping = EarlyStopping(monitor="val_accuracy",
                              patience = 3,
                              restore_best_weights = True)

#training
history = model.fit(x_train,y_train,
                    epochs = 10,
                    batch_size = 16,
                    validation_split = 0.2,
                    callbacks = [early_stopping])


#model test ve evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, test accuracy:{accuracy}")

#training ve validation loss görselleştirmesi
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['loss'],marker = "o", label='Training Loss')
plt.plot(history.history['val_loss'],marker = "o", label='Validation Loss')
plt.title("LSTM loss on IMDB Dataset")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#training ve validation accuracy görselleştirmesi
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],marker = "o", label='Training Accuracy')
plt.plot(history.history['val_accuracy'],marker = "o", label='Validation Accuracy')
plt.title("LSTM accuracy on IMDB Dataset")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
