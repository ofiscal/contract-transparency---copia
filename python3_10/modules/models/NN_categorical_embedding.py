# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:46:40 2023

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

def create_model_categorical(
        categorical_vars:pd.DataFrame()
        )->tf.keras.Model:
    #this are the parameters for the model, we will update them as needed
    argumentos
    inputs = layers.Input(shape=(argumentos.max_length,))

    model=tf.keras.Sequential([
        tf.keras.layers.Embedding(argumentos.vocab_size,
                                  argumentos.embedding_dim, 
                                  input_length=argumentos.max_length),#reduce word dimesionality
        tf.keras.layers.Dense(argumentos.embedding_dim, activation='softmax'),#normal layer neurons
        tf.keras.layers.Dense(argumentos.embedding_dim, activation='softmax'),#normal layer neurons
        tf.keras.layers.Dense(1)#one outcome as is a regressión
    ])

    model.compile(optimizer=Adam(learning_rate=argumentos.learning_rate, decay=argumentos.decay),
    loss='mean_absolute_error',
    metrics=["KLDivergence","MeanSquaredError"])
    return model


def categorical_train(    # TODO : These default arguments should probably
    # both be defined from `data`,
    # rather than one of them from `datatest`.
    X: pd.Series,
    Y:pd.Series,
    checkpointpath,
    )->(tf.keras.Sequential,
    tf.keras.preprocessing.text.Tokenizer,float,float):
    

    
    tokenizer= Tokenizer(num_words=argumentos.vocab_size,oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(X)
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=argumentos.max_length)
    labels2=Y.values
    labelsmean=labels2.mean()
    labelssd=labels2.std()
    labels=(labels2-labelsmean)/labelssd
    


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointpath,
        save_weights_only=True,
        mode='max',
        save_freq=200)
    model=create_model_categorical()


    model.fit(padded,labels,epochs=argumentos.num_epochs,validation_split=0.2,verbose=2,callbacks=[model_checkpoint_callback])
    return model,tokenizer,labelsmean,labelssd
    ##
    
def keep_train(model:tf.keras.Model,
    epocas:float,
    X: pd.DataFrame,
    Y:pd.Series,
    checkpointpath,
    mean:float,
    stdv:float
    )->tf.keras.Model:
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointpath,
        save_weights_only=True,
        mode='max',
        save_freq=200)
    tokenizers=list()
    for i in X:
        tokenizer= Tokenizer(num_words=argumentos.vocab_size,oov_token="<OOV>")
        tokenizer.fit_on_texts(i)
        tokenizers.append(tokenizer)

    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(X)
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=argumentos.max_length)
    labels2=Y.values
    labelsmean=mean
    labelssd=stdv
    labels=(labels2-labelsmean)/labelssd
    
    model.fit(padded,labels,epochs=epocas,validation_split=0.2,verbose=2,callbacks=[model_checkpoint_callback])
    return model
    