#this are models for recurrent neural networks and nlp.
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from typing import Tuple




def recurrent_train (
    # TODO : These default arguments should probably
    # both be defined from `data`,
    # rather than one of them from `datatest`.
    X: pd.Series,
    Y:pd.Series
    )->(tf.keras.Sequential,
             tf.keras.preprocessing.text.Tokenizer,float,float):
    #this are the parameters for the model, we will update them as needed
    vocab_size=100000
    embedding_dim=100
    max_length=200
    num_epochs=1
    learning_rate=0.001
    decay=0.00001
    #we prepare the information for the model
    #we generate tokens for every word in the X descriptions
    tokenizer= Tokenizer(num_words=vocab_size,oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    word=tokenizer.word_index
    #we generate series from the description text using the tokens instead of word
    sequences=tokenizer.texts_to_sequences(X)
    #we padd them to make the sequences of equal length
    padded=pad_sequences(sequences,maxlen=max_length)
    #we normalize the data using the standar deviation and the mean
    labels2=Y.values
    labelsmean=labels2.mean()
    labelssd=labels2.std()
    labels=(labels2-labelsmean)/labelssd
    #neural network structure
    model1=tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),#reduce word dimesionality
        Bidirectional(LSTM(embedding_dim)),#recurrent long short memory
        tf.keras.layers.Dense(embedding_dim, activation='softmax'),#normal layer neurons
        tf.keras.layers.Dense(1)#one outcome as is a regressión
    ])

    model1.compile(optimizer=Adam(learning_rate=0.001, decay=0.00001),
        loss='mean_absolute_error',
        metrics=["KLDivergence","MeanSquaredError"])

    model1.fit(padded,labels,epochs=num_epochs,validation_split=0.2,verbose=2)
    return model1,tokenizer,labelsmean,labelssd