# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:46:40 2023

@author: usuario
"""
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


def create_model_categorical(
        )->tf.keras.Model:
    #this are the parameters for the model, we will update them as needed
    argumentos
    inputs = layers.Input(shape=(argumentos.max_length,))
    
    embedding_layer = TokenAndPositionEmbedding(argumentos.max_length, argumentos.vocab_size, argumentos.embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(argumentos.embedding_dim, argumentos.num_heads, argumentos.ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=argumentos.learning_rate, decay=argumentos.decay),
    loss='mean_absolute_error',
    metrics=["KLDivergence","MeanSquaredError"])
    return model
