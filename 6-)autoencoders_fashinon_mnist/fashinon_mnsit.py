import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam

(x_train, _ ), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

plt.figure()
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
    
plt.show()

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

input_dim = x_train.shape[1]
encoding_dim = 64

input_image = Input(shape = (input_dim,))
encoded = Dense(256, activation="relu")(input_image)
encoded = Dense(128, activation="relu")(encoded)
encoded = Dense(encoding_dim, activation="relu")(encoded)

decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(256, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_image, decoded)

autoencoder.compile(optimizer = Adam(), loss = "binary_crossentropy")

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(x_test,x_test),
    verbose = 1
    )

encoder = Model(input_image, encoded)

encoded_input = Input(shape = (encoding_dim,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_output = autoencoder.layers[-1](decoder_layer2)

decoder = Model(encoded_input, decoder_output)

encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)
n = 10

plt.figure(figsize = (15,5))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_images[i].reshape(28,28),cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed)**2)
    if mse ==0:
        return float("inf")
    
    max_pixel = 1.0
    psnr = 20*np.log10(max_pixel/np.sqrt(mse))
    return psnr

psnr_score = []

for i in range(100):
    original_img = x_test[i]
    reconstructed_img = decoded_images[i]
    score = calculate_psnr(original_img, reconstructed_img)
    psnr_score.append(score)
    
average_psnr = np.mean(psnr_score)
print(f"average_psnr: {average_psnr}")