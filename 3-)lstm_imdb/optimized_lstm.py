# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 01:33:23 2025

@author: cirim
"""

"""

LSTM modeliniz için hipermetre optimizasyonu yapmak adına, bazı hiperparametreleri iyileştirmeniz 
gerekebilir. Bu optimizasyonlar arasında, modelin katman yapısı, öğrenme oranı, dropout oranı, batch size,
 LSTM hücre sayısı, vb. yer alır. Ayrıca, Keras Tuner veya GridSearchCV gibi yöntemleri kullanarak 
 modelin farklı parametre kombinasyonlarını test edebilirsiniz. İşte modelinize yönelik önerilen bazı 
 iyileştirmeler:

1. LSTM Katmanının Hücre Sayısı (Units)
LSTM katmanındaki units parametresi, modelin öğrenme kapasitesini etkileyen önemli bir faktördür. 
Bu değeri optimize etmek, modelin daha iyi performans göstermesini sağlayabilir. Daha yüksek units 
değeri genellikle daha iyi performans sağlar, fakat fazla büyük değerler aşırı öğrenmeye (overfitting) 
yol açabilir.

2. Dropout Oranı
Dropout oranı, aşırı öğrenmeyi (overfitting) engellemek için önemlidir. Ancak, çok yüksek bir dropout 
oranı, modelin doğru genelleme yapmasını zorlaştırabilir. Oranın iyileştirilmesi faydalı olabilir.

3. Optimizer ve Öğrenme Oranı (Learning Rate)
Farklı optimizasyon algoritmalarını ve öğrenme oranlarını deneyerek modelinizin daha hızlı ve doğru 
bir şekilde öğrenmesini sağlayabilirsiniz.

4. Epoch ve Batch Size
Epoch sayısı ve batch size, eğitim süresi ve modelin performansı üzerinde önemli bir etkiye sahiptir. 
Çok büyük batch size’lar genellikle daha hızlı eğitim sağlar, ancak küçük batch size’lar daha stabil 
bir öğrenme sağlayabilir. Epoch sayısını da optimal hale getirebilirsiniz.

5. Modelin Katman Sayısı ve Türü
Birden fazla LSTM katmanı kullanmak veya farklı türde katmanlar (örneğin, GRU veya Bidirectional LSTM) 
eklemek, modelin başarısını artırabilir.

Örnek Hipermetre Optimizasyonu:
Aşağıdaki kodda, birkaç hipermetreyi optimize edebilmek için Keras Tuner ile GridSearch yöntemini 
kullanacağız. Bu şekilde en uygun parametre kombinasyonlarını bulabilirsiniz.
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Veriyi yükle ve hazırla
max_features = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Modeli oluşturma fonksiyonu
def build_lstm_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=hp.Int('embedding_output_dim', 
                                                                  min_value=32, 
                                                                  max_value=128, 
                                                                  step=32), input_length=maxlen))
    
    # LSTM katmanındaki hücre sayısı (units) optimize ediliyor
    model.add(LSTM(units=hp.Int('lstm_units', min_value=16, max_value=128, step=16), 
                   dropout=0.2, recurrent_dropout=0.2))
    
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 
                                                        min_value=1e-5, 
                                                        max_value=1e-3, 
                                                        sampling='LOG', 
                                                        default=1e-4)),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    return model

# Hiperparametre arayüzü
hp = HyperParameters()

# Hyperparameter tuning (RandomSearch)
tuner = RandomSearch(
    build_lstm_model,
    objective="val_accuracy",
    max_trials=5,  # Bu sayıyı arttırarak daha fazla deneme yapabilirsiniz
    executions_per_trial=1,
    directory="my_dir",
    project_name="imdb_lstm_tuning",
    overwrite=True
)

# EarlyStopping
early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

# Eğitim ve tuning
tuner.search(x_train, y_train,
             epochs=10,
             batch_size=16,
             validation_split=0.2,
             callbacks=[early_stopping])

# En iyi model
best_model = tuner.get_best_models(num_models=1)[0]

# Modelin özetini yazdır
best_model.summary()

# Test seti üzerinde değerlendirme
test_loss, test_acc = best_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

"""
Yaptığım İyileştirmeler:
Embedding katmanındaki output_dim: Farklı değerler denemek için, embedding boyutunu hp.Int ile 
optimizasyon yaptım.
LSTM units sayısı: LSTM hücre sayısını da hp.Int ile optimize ediyorum.
Learning rate: Öğrenme oranını hp.Float ile logaritmik olarak optimize ediyorum.
Keras Tuner ile Parametre Aramaları:
Bu optimizasyon yöntemi, daha geniş bir parametre alanında arama yaparak, en iyi model 
parametrelerini bulmanıza yardımcı olabilir. Ayrıca, RandomSearch yerine Hyperband veya 
BayesianOptimization gibi başka optimizasyon yöntemleri de kullanabilirsiniz.

Bu önerilen adımlar ve iyileştirmelerle modelinizin doğruluğunu artırabilirsiniz.
"""

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

