# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:01:39 2023

@author: Daniel Duque
"""
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from python3_10.modules.models.models_main import argumentos
from python3_10.modules.models.Transformers import TransformerBlock,TokenAndPositionEmbedding
import numpy as np
import python3_10.modules.cleaning as cleaning
from sklearn.metrics import r2_score
import json
import io
import os
from keras.utils.vis_utils import plot_model
import sys
import mlflow
#the structural definition of the transformer block and tokens was taken from
#https://keras.io/examples/nlp/text_classification_with_transformer/ with 
#some changes



def transformer_layer(inputsx:layers.Input,
                      x
                      ):
    
    
    embedding_layer = TokenAndPositionEmbedding(argumentos.max_length,
                                                argumentos.vocab_size,
                                                argumentos.embedding_dim)
    x = embedding_layer(inputsx)
    transformer_block = TransformerBlock(argumentos.embedding_dim,
                                         argumentos.num_heads,
                                         argumentos.ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(200, activation="relu",kernel_regularizer="l1_l2", use_bias=False)(x)
    x=tf.keras.layers.Dropout(.2, input_shape=(200,))(x)
    x = layers.Dense(70, use_bias=False)(x)
    return keras.Model(inputs=inputsx, outputs=x)



#this will be later transplanted to the main file, but for now I need to test
#the code somewhere and tdidn´t find a better place
#some paths to where things are
path_to_models=r"trainedmodels"
pathToData = r"data/sucio/CONTRATOS_COVID(20-22).csv"
path_to_result=r"data/resultados"
#this cant be change, they are how rn where train (will be changed manually)



mean=163740447
ssd=1716771980
#loading the cleaned dataset
data_pred=cleaning.secop_for_prediction(pathToData=pathToData)

#entrenar los modelos o continuar su entrenamiento
#we select from "TR" for trasformer "RN" for recurrent "NN" for neural network
#
entrenar="TR" 
setsize=400
data_desc=cleaning.secop2_general(pathToData =pathToData,subsetsize=setsize)
data_categ2=cleaning.secop2_categoric(pathToData =pathToData,subsetsize=setsize)
data_categ2=data_categ2.astype(str).applymap(lambda x:[x.replace(" ","")])

data_categ=pd.DataFrame()
#generamos un tokenizer por cada categoria y lo aplicamos, mantener en falso
if True:
    for column in data_categ2:
        try:        
            tokenizer2= Tokenizer(num_words=argumentos.vocab_size,oov_token="<OOV>")
            tokenizer2.fit_on_texts(data_categ2[column])
            #we generate series from the description text using the tokens instead of word

            data_categ[column]=data_categ2[column].apply(lambda x:int(tokenizer2.texts_to_sequences(x)[0][0]))

            tokenizer_json2=tokenizer2.to_json()
            pathto_token=r"trainedmodels\tokenizers"+"\\"+column+"tokenizer.json"
                     
            try:
                os.mknod(pathto_token)  
            except:
                ...
            with io.open(pathto_token,"w",encoding="utf-8") as f:
                f.write(json.dumps(tokenizer_json2,ensure_ascii=False))
            print(column)

            
        except KeyboardInterrupt:
            print("variable "+ column+" was not found")
    

    
    
    tokenizer= Tokenizer(num_words=argumentos.vocab_size,oov_token="<OOV>")
    tokenizer.fit_on_texts(data_desc['detalle del objeto a contratar'].astype(str))
    tokenizer_json=tokenizer.to_json()
    pathto_token=r"trainedmodels\tokenizers"+"\\"+"desctokenizer.json"
    with io.open(pathto_token,"w",encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json,ensure_ascii=False))

for column in data_categ2:
    try:

        pathto_token=r"trainedmodels\tokenizers"+"\\"+column+"tokenizer.json"
        f = open(pathto_token)
        tokenizer2=tokenizer_from_json(json.load(f))
        data_categ[column]=data_categ2[column].apply(lambda x:int(tokenizer2.texts_to_sequences(x)[0][0]))
    except:
        ...


data_value=cleaning.secop2_valor(pathToData =pathToData,subsetsize=setsize).apply(
        lambda x:(x-mean)/ssd)
tokenizer= Tokenizer(num_words=argumentos.vocab_size,oov_token="<OOV>")
f = open(r"trainedmodels\tokenizers"+"\\"+"desctokenizer.json")
tokenizer=tokenizer_from_json(json.load(f))
#we generate series from the description text using the tokens instead of word
sequences=tokenizer.texts_to_sequences(data_desc['detalle del objeto a contratar'].astype(str))
#we padd them to make the sequences of equal length
padded=pad_sequences(sequences,maxlen=argumentos.max_length)
#transform the data in np array



#model=create_model_full(categorical_vars=data_categ,
#          transformer_vars=padded)


model,inputvar=full_train(categorical_vars=data_categ,transformer_vars=padded,
                          output=data_value,checkpointpath=path_to_models+r"\modelfull_tr.hdf5",
                          load=True,fit=False)



predict=pd.DataFrame(model.predict(inputvar))

data_value["predict"]=predict



data_value.to_excel(path_to_result+r"\results3.xlsx")


r2_score(data_value["valor norm"],data_value["predict"])


