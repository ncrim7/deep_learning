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

word_index = imdb.get_word_index()

#kelime dizinini geri döndürmek için geri çevirelim
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>" #pad ile eşleştir
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

#örnek metinleri yazdırma
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review])

#rastgele 3 örnek yazdıralım
random_indices = np.random.choice(len(x_train), size = 3, replace = False)

for i in random_indices:
    print(f"Yorum: {decode_review(x_train[i])}")
    print(f"Etiket: {y_train[i]}")
    print()
    
#transformer mimarisindeki bir blok yapı sınıfı
#bu blok self-attention ve feed forward ağını birleştirecek
"""class TransformerBlock(layers.Layer):
    
    #embed_size: girişteki emdedding vektörlerinin boyutu
    #heads: head attention mekanizmasında kullanılacak olan başlık sayısı
    #dropout_rate: ağırlıkların sıfırlanma oranı overfittingi engelleme yöntemlerinden biri
    def __init__(self, embed_size, heads, dropout_rate=0.3):
        #üst sınıfımız olan layers.layer sınıfının init metodunu çağırarak temel katman özelliklerini aktarma
        super(TransformerBlock, self).__init__()
        
        #multi head dikkat mekanizması
        #num_heads: başlık sayısı (aynı anda kaç farklı dikkat hesaplaması yapacağımızı belirleyen parametre)
        #key_dim: her dikkak baaşlığında kullanılan anahtar boyutu
        self.attention = layers.MultiHeadAttention(num_heads = heads, key_dim=embed_size)
        
        #normalizasyon katmanı: epsilon sayısal kararlılık artırmak için kullanılır.
        self.norm1 = layers.LayerNormalization(epsilon = 1e-6)
        self.norm2 = layers.LayerNormalization(epsilon = 1e-6)
        
        self.feed_forward = models.Sequential([
            layers.Dense(embed_size * heads, activation = "relu"),
            layers.Dense(embed_size)]) #çıkışı tekrar orjinal embed sayısına getirir
        
        #overfittingi engellemek için kullanılan katman
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    #call: girişin nasıl işlendiğini tanımla yani transformer işlevi
    #x: giriş vektörümüz
    #training: eğitim sırasında dropout uygulayıp uygulanmamasını belirler
    def call(self, x, training):
        #self attention mekanizması: her bir kelimenin diğer kelimelerle bağlam ilişkisini öğreni
        attention = self.attention(x,x)
        x=self.norm1(x+self.dropout1(attention, training = training))
        feed_forward = self.feed_forward(x)
        return self.norm2(x+self.dropout2(feed_forward, training=training))"""
    
    
# TransformerBlock sınıfını düzeltelim
class TransformerBlock(layers.Layer):  # 'layer' yerine 'Layer' olmalı
    
    def __init__(self, embed_size, heads, dropout_rate=0.3):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention mechanism
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        
        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed forward network
        self.feed_forward = models.Sequential([
            layers.Dense(embed_size * heads, activation="relu"),
            layers.Dense(embed_size)])  # çıkışı tekrar orjinal embed boyutuna getirir
        
        # Dropout layers
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, x, training):
        # Self-attention mechanism
        attention = self.attention(x, x)
        x = self.norm1(x + self.dropout1(attention, training=training))
        
        # Feed-forward network
        feed_forward = self.feed_forward(x)
        
        # Final normalization
        return self.norm2(x + self.dropout2(feed_forward, training=training))
    
# TransformerModel sınıfını düzeltelim
class TransformerModel(models.Model):
    
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding katmanı
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=embed_size)
        
        # Transformer blokları
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout_rate) for _ in range(num_layers)]
        
        # Global Average Pooling
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        
        # Dropout katmanı
        self.dropout = layers.Dropout(dropout_rate)
        
        # Fully connected (FC) katmanı
        self.fc = layers.Dense(output_dim, activation="sigmoid")
        
    def call(self, x, training):
        # Embedding katmanından geçir
        x = self.embedding(x)
        
        # Transformer blokları
        for transformer in self.transformer_blocks:
            x = transformer(x, training)
        
        # Global Average Pooling
        x = self.global_avg_pooling(x)
        
        # Dropout uygulama
        x = self.dropout(x, training=training)
        
        # Çıkış katmanı
        return self.fc(x)


# Modelin parametrelerini ayarlayalım
num_layers = 4
embed_size = 64
num_heads = 4
input_dim = max_features
output_dim = 1  # Binary classification için 1 çıktı
dropout_rate = 0.1

# Modeli oluştur
model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate)

# Modeli derleyelim
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Model özeti
model.build(input_shape=(None, maxlen))
model.summary()

history = model.fit(x_train, y_train, epochs = 2, batch_size = 256, validation_data = (x_test, y_test))

#model test ve evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, test accuracy:{accuracy}")

#training ve validation loss görselleştirmesi
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history['loss'],marker = "o", label='Training Loss')
plt.plot(history.history['val_loss'],marker = "o", label='Validation Loss')
plt.title("Transformers loss on IMDB Dataset")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#training ve validation accuracy görselleştirmesi
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],marker = "o", label='Training Accuracy')
plt.plot(history.history['val_accuracy'],marker = "o", label='Validation Accuracy')
plt.title("Transformers accuracy on IMDB Dataset")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#kullanıcının yazdığı metinin tahmin edilmesi
def predict_sentiment(model, text, word_index, maxlen):
    #metni sayısal formata çevir
    encoded_text = [word_index.get(word,0) for word in text.lower().split()] #kelimeleri sayıya çevirir
    #padding
    padded_text = pad_sequences([encoded_text], maxlen = maxlen)
    #prediction = yorumun olumlu mu olumsuz mu olduğunun tahmin edilmesi
    prediction = model.predict(padded_text) #model ile tahmin yap
    return prediction[0][0]
#imdb veri setindeki kelime dizisi
word_index = imdb.get_word_index()

#kullanıcıdan metin al
user_input = input("Bir film yorumu yazın: ")
sentiment_score = predict_sentiment(model, user_input, word_index, maxlen)
print(sentiment_score)

if sentiment_score > 0.5:
    print(f"Tahmin Sonucu %{int(round(sentiment_score*100,0))} olasiliği ile olumlu skor: {sentiment_score}")
else:
    print(f"Tahmin Sonucu %{100 - int(round(sentiment_score*100,0))} olasiliği ile olumsuz skor: {sentiment_score}")

"""   
#Transformers Model Oluşturma
class TransformerModel(models.Model):
    
    #num_layers: transfromer bloklarının sayısı
    #embed_size: girdi verilerinin embedding boyutu
    #heads: herbirtransformer bloğundaki multi head attention sayısı
    #input_dim: girişteki olası token sayısı
    #output_dim: modelin çıktısındaki sınıf boyutu
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate = 0.1):
        super(TransformerModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout_rate) for _ in range (num_layers)]
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(output_dim, activation = "sigmoid")
        
    def call(self, x, training):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, training)
            x = self.global_avg_pooling(x)
            x = self.dropout(x, training = training)
            return self.fc(x)
        
num_layers = 4
embed_size = 64
num_heads = 4
input_dim = max_features
output_dim = 1
dropout_rate = 0.1

model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate)
        
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.build(input_shape = (None, maxlen))      
model.summary()"""