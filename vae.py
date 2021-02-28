import keras
from keras import layers
from keras import backend as K
from keras.datasets import mnist
from keras.models import model_from_json

from sklearn.model_selection import train_test_split
import numpy as np

import os.path
from os import path

import matplotlib.pyplot as plt
import pickle

import hdbscan

# GLOBAL PARAMETERS
original_dim = 100 # feature dimensions of data
intermediate_dim = 50
latent_dim = 2 # Layer used for analysis

batch_size = 42
epochs = 5

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

if __name__ == '__main__':
    # Define model layers
    inputs = keras.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_sigma = layers.Dense(latent_dim)(h)

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Implement VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    # Train on data
    X = pickle.load(open('data/data.pkl', 'rb'))
    x_train, x_test = train_test_split(X, test_size=0.42, random_state=42)

    # Load pre-trained VAE if it exists
    if not path.exists("models/en_model.h5"):
        vae.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

        # Save model specs
        model_json = vae.to_json()
        with open("models/en_model.json", "w") as json_file:
            json_file.write(model_json)
        # Serialize trained weights to HDF5
        vae.save_weights("models/en_model.h5")
    else:
        # Load model specs
        json_file = open('models/en_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        vae = model_from_json(loaded_model_json)
        # Load weights into new model
        vae.load_weights("models/en_model.h5")

    # Visualize embedding
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
    
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(x_test_encoded)
    labels = clusterer.labels_
    print('Labels:', set(labels))

    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=labels)
    plt.colorbar()
    plt.savefig('plots/en_2d.png')
    plt.close()