# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:01:39 2023

@author: Daniel Duque
"""
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from python_3_10.modules.models.models_main import argumentos
from python_3_10.modules.models.Transformer import TransformerBlock,TokenAndPositionEmbedding
import numpy as np
#the structural definition of the transformer block and tokens was taken from
#https://keras.io/examples/nlp/text_classification_with_transformer/ with 
#some changes



def transformer_layer(inputsx:layers.Input
                      ):
    embedding_layer = TokenAndPositionEmbedding(argumentos.max_length, argumentos.vocab_size, argumentos.embedding_dim)
    x = embedding_layer(inputsx)
    transformer_block = TransformerBlock(argumentos.embedding_dim, argumentos.num_heads, argumentos.ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    return x


def categorical_layer(inputsy:layers.Input,
                      categorical_vars:pd.DataFrame()
                      ):
    inputss = []
    embeddings = []
    for c in categorical_vars:
        inputs = layers.Input(shape=(1,),name='input_sparse_'+c)

        embedding = TokenAndPositionEmbedding(argumentos.max_word,
                                              argumentos.vocab_size,
                                              argumentos.embedding_dim)
        inputss.append(inputs)
        embeddings.append(embedding)
    y=layers.concatenate()(embeddings)
    y = layers.Dense(100, activation="relu")(y)
    y=layers.Dropout(0.5)(y)
    y=layers.Dense(1)(y)
    return y

def numerical_layer(inputsz:pd.DataFrame(),
                    numerical_vars:pd.DataFrame()
                    ):
    z= layers.Dense(len(inputsz.columns))(inputsz)
    z = layers.Dense(100, activation="relu")(z)
    z = layers.Dropout(0.5)(z)
    outputs = layers.Dense(1)(z)
def create_model_full(numeric_var:int,
                      categorical_vars:pd.DataFrame(),
                      numerical_vars:pd.DataFrame()
        ):
    argumentos
    inputsx = layers.Input(shape=(argumentos.max_length,))
    inputsy = layers.Input(shape=(argumentos.max_word,))
    inputsz=layers.Input(shape=(len(numerical_vars),))
    input_numeric = layers.Input(shape=(argumentos.max_word,))
    ##Transformer layer
    transf=transformer_layer(inputsx=inputsx)
    categor=categorical_layer(inputsy=inputsy,
                              categorical_vars=categorical_vars)
    ##




    
    
    
    combined = concatenate([x.output, y.output])
    
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    
def full_train(
                categorical_vars:pd.DataFrame(),
        numerical_vars:pd.DataFrame(),
        transformer_vars:pd.DataFrame(),
        conv_vars:pd.DataFrame()
        )->tf.keras.Model:
    ...
        