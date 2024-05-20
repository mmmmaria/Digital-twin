import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Dense, Reshape, Flatten, LeakyReLU,
    LayerNormalization, Dropout, BatchNormalization
    )

def build_generator(latent_space, n_var, n_features=2,use_bias=True):

    model = tf.keras.Sequential()
    model.add(Dense(15, input_shape=(latent_space,), activation='relu', use_bias=use_bias))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu'))                  # 10
    model.add(BatchNormalization())
    model.add(Dense((20), activation='relu'))          # 25
    model.add(BatchNormalization())
    model.add(Dense(n_features, activation="tanh", use_bias=use_bias))

    return model

def build_critic(n_var, use_bias=True):

    model = tf.keras.Sequential()
    model.add(Dense(25, input_shape=(n_var,), use_bias=use_bias))
    # model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1))

    return model

# 15, 5, 5 (generator for circle)

#3d dip data.
# def build_generator(latent_space, n_var):
#
#     model = tf.keras.Sequential()
#     model.add(Dense(n_var*15, input_shape=(latent_space,), use_bias=True))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Dense(n_var*5, use_bias=True))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     model.add(Dense(n_var*5, use_bias=True))
#     model.add(Dense(n_var, activation="tanh", use_bias=True))
#
#     return model
#
# def build_critic(n_var):
#
#     model = tf.keras.Sequential()
#     model.add(Dense(n_var*5, use_bias=True))
#     model.add(LayerNormalization())
#     model.add(LeakyReLU())
#     model.add(Dropout(0.2))
#     model.add(Dense(n_var*15))
#     model.add(LayerNormalization())
#     model.add(LeakyReLU())
#     model.add(Flatten())
#     model.add(Dense(1))
#
#     return model
