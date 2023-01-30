# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:09:56 2023

@author: Daniel Duque
"""
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

#the structural definition of the transformer block and tokens was taken from
#https://keras.io/examples/nlp/text_classification_with_transformer/ with 
#some changes


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.05):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def create_model_tr(
        )->tf.keras.Model:
    #this are the parameters for the model, we will update them as needed
    vocab_size=100000
    embedding_dim=100 #this is the dimension that vocabulary will be reduced
    max_length=200 #length of the sentences
    num_epochs=7
    learning_rate=0.001
    decay=0.00001
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    inputs = layers.Input(shape=(max_length,))
    
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate, decay=decay),
    loss='mean_absolute_error',
    metrics=["KLDivergence","MeanSquaredError"])
    return model

def transformer_train(    # TODO : These default arguments should probably
    # both be defined from `data`,
    # rather than one of them from `datatest`.
    X: pd.Series,
    Y:pd.Series,
    checkpointpath,
    )->(tf.keras.Sequential,
    tf.keras.preprocessing.text.Tokenizer,float,float):
    
    vocab_size=100000
    embedding_dim=100 #this is the dimension that vocabulary will be reduced
    max_length=200 #length of the sentences
    num_epochs=7
    learning_rate=0.001
    decay=0.00001
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    
    tokenizer= Tokenizer(num_words=vocab_size,oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(X)
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=max_length)
    labels2=Y.values
    labelsmean=labels2.mean()
    labelssd=labels2.std()
    labels=(labels2-labelsmean)/labelssd
    


    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointpath,
        save_weights_only=True,
        mode='max',
        save_freq=200)
    model=create_model_tr()


    model.fit(padded,labels,epochs=num_epochs,validation_split=0.2,verbose=2,callbacks=[model_checkpoint_callback])
    return model,tokenizer,labelsmean,labelssd

def keep_train(model:tf.keras.Model,
    epocas:float,
    X: pd.Series,
    Y:pd.Series,
    checkpointpath
    )->tf.keras.Model:
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointpath,
        save_weights_only=True,
        mode='max',
        save_freq=200)
    
    vocab_size=100000
    embedding_dim=100 #this is the dimension that vocabulary will be reduced
    max_length=200 #length of the sentences
    num_epochs=7
    learning_rate=0.001
    decay=0.00001
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

 
    
    tokenizer= Tokenizer(num_words=vocab_size,oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(X)
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=max_length)
    labels2=Y.values
    labelsmean=labels2.mean()
    labelssd=labels2.std()
    labels=(labels2-labelsmean)/labelssd
    
    model.fit(padded,labels,epochs=epocas,validation_split=0.2,verbose=2,callbacks=[model_checkpoint_callback])
    return model
    
    
    
    
    
    
    
    
    
    
    
