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
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    return x


def categorical_layer(inputy:layers.Input,
                      categorical_vars:pd.DataFrame()
                      ):
    inputss = []
    embeddings = []
    for c in categorical_vars:
        inputs = layers.Input(shape=(1,),name='input_sparse_'+c)

        embedding = Embedding(argumentos.max_word, embedding_size, input_length = 1)(inputs)
        embedding = Reshape(target_shape=(embedding_size,))(embedding)
        inputss.append(inputs)
        embeddings.append(embedding)
    input_numeric = Input(shape=(1,),name='input_constinuous')
    embedding_numeric = Dense(16)(input_numeric)
    inputss.append(input_numeric)
    embeddings.append(embedding_numeric)
    y = Concatenate()(embeddings)

def create_model_full(
        ):
    argumentos
    inputsx = layers.Input(shape=(argumentos.max_length,))
    inputsy = layers.Input(shape=(argumentos.max_word,))
    ##Transformer layer
    transf=transformer_layer(inputsx=inputsx)

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
        