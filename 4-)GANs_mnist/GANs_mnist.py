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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten,Conv2D, Conv2DTranspose, BatchNormalization 
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img, array_to_img
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, _ ), (_ , _ ) = mnist.load_data()

#normalization
x_train = x_train /255.0

#boyutların (28,28) den (28,28,1) e dönüştürülmesi
x_train = np.expand_dims(x_train, axis = -1)

# discriminator modeli tanımla = discriminator gerçek mi fake mi karşılaştırma yapar
def build_discriminator():
    
    model = Sequential()
    # feature extraction
    # 64 filtre (3x3)
    model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same", input_shape = (28,28,1)))
    model.add(LeakyReLU(alpha = 0.2))
    
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = "same"))
    model.add(LeakyReLU(alpha = 0.2))
    
    #classification fake mi yoksa reel mi
    model.add(Flatten()) 
    model.add(Dense(1,activation = "sigmoid"))
    
    model.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5),metrics = ["accuracy"] )
    model.summary()
    return model
build_discriminator()
zdim = 100
#generator modeli tanımla = generator fake veriler üretir
def build_generator():
    model  = Sequential()
    
    model.add(Dense(7*7*128, input_dim = zdim)) #gürültü vektöründen yüksek boyutlu uzaya dönüşüm
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Reshape((7,7,128)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="tanh"))
    
    return model
build_generator().summary()

#discriminator ve generator kullanarak GAN modeli oluştur.
def build_gan(generator, discriminator):
    discriminator.trainable = False #discriminator GAN içerisinde eğitilmez
    
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss="binary_crossentropy", optimizer= Adam(0.0002, 0.5))
    
    return model
    
discriminator = build_discriminator()
generator = build_generator()
gan = build_gan(generator, discriminator)
gan.summary()

#gan training
epochs = 10000
batch_size = 128
half_batch = batch_size//2

#eğitim döngüsü
for epoch in tqdm(range(epochs), desc = "Training Process"): #tqdm ilerlerme çubuğu
    #fake veriler ve gerçek veriler ile discriminator eğitimi
    #gerçek veriler ile discriminator eğitimi
    idx = np.random.randint(0, x_train.shape[0], half_batch) #x_train içerisinden rastgele32
    real_images = x_train[idx] #gerçek görüntüler
    real_labels = np.ones((half_batch, 1)) #gerçek görüntü etiketi = 1
    
    #fake verileri (generatorun ürettiği) kullanarak discriminator eğitimi
    noise = np.random.normal(0,1, (half_batch, zdim)) #gürültü vektörü oluştu
    fake_images = generator.predict(noise, verbose = 0) #generator ile üretilen görüntü
    fake_labels = np.zeros((half_batch,1)) #sahte görüntü etiketleri = 0
                           
    #update discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels) #gerçek görüntüler ile loss 
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels) #fake görünler ile loss
    d_loss = np.add(d_loss_fake, d_loss_real) * 0.5 #ortalama kayıp
    
    #gan eğitimi
    noise = np.random.normal(0,1, (batch_size, zdim)) #gürültü oluştur
    valid_y = np.ones((batch_size, 1)) #doğru etiketler
    g_loss = gan.train_on_batch(noise, valid_y) #gan in içinde bulunan generatorun eğitimi
    
    if epoch % 100 == 0:
        print(f"{epoch}/{epochs} d_loss: {d_loss[0]}, g_loss: {g_loss}")


def plot_generated_images(generator, examples = 10, dim = (1,10)):
    noise = np.random.normal(0,1, (examples,zdim)) #gürültü vektörleri
    gen_images = generator.predict(noise, verbose = 0)#üretilen görüntülerimiz
    gen_images = 0.5 * gen_images + 0.5

    plt.figure(figsize = (10,1))
    for i in range(gen_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(gen_images[i, :, :, 0], cmap="gray") #görüntüyü gri tonlamasında göster
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()
    
plot_generated_images(generator)
        
        
#modeli kaydetme
gan.save("final_gan_model.keras")