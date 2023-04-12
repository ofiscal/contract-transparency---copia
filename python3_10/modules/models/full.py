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

    x = layers.Dense(2000, activation="relu",kernel_regularizer="l1_l2")(x)

    x = layers.Dense(700)(x)
    return keras.Model(inputs=inputsx, outputs=x)


def categorical_layer(inputsy:layers.Input,
                      categorical_vars:pd.DataFrame()
                      )->keras.layers:
    inputss = []
    embeddings = []

    for c in categorical_vars:

        inputs = layers.Input(shape=(1,),name='input_sparse_'+c)
        print('input_sparse_'+c)
        
        embedding = TokenAndPositionEmbedding(argumentos.max_word,
                                              argumentos.vocab_size,
                                              argumentos.embedding_dim)
        inputss.append(inputs)
        embeddings.append(embedding)

    #input_numeric = keras.Input(shape=(1,),name='input_continuous')
    #embedding_numeric = layers.Dense(16)(input_numeric)
    #embeddings.append(embedding_numeric)
    #inputss.append(input_numeric)
    y=layers.Concatenate()(inputss)
    y = layers.Dense(2000, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.2, l2=0.2))(y)

    y=layers.Dense(700)(y)
    return keras.Model(inputs=inputss, outputs=y)

def numerical_layer(inputsz:pd.DataFrame(),
                    numerical_vars:pd.DataFrame()
                    ):
    z= layers.Dense(len(inputsz.columns))(inputsz)
    z = layers.Dense(100, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.2, l2=0.2))(z)

    z = layers.Dense(10)(z)
    return keras.Model(inputs=inputsz, outputs=z)
    
def create_model_full(categorical_vars:pd.DataFrame(),

                      transformer_vars:pd.DataFrame(),
        ):
    argumentos
    inputsx = layers.Input(shape=(argumentos.max_length,))
    inputsy = layers.Input(shape=(len(categorical_vars.columns),))

  
    #the next layers are paralelized
    
    #categorical layer
    categor=categorical_layer(inputsy=inputsy,
                              categorical_vars=categorical_vars)
    ##Transformer layer
    transf=transformer_layer(inputsx=inputsx,x=transformer_vars)

    #numeric layer

    ##

    print(categor)
    print(transf)
    combined = layers.concatenate([ categor.output,transf.output])
    ka = layers.Dense(50,
                      kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.2, l2=0.2))(combined)
    ka = layers.Dense(1)(ka)
    
    model = keras.Model(inputs=[categor.input,transf.input],
                        outputs=ka)
    return model
    
def full_train(
                categorical_vars:pd.DataFrame(),
        transformer_vars:pd.DataFrame(),
        output:pd.DataFrame(),
        checkpointpath,
        load,
        fit=True,

        )->tf.keras.Model:
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointpath,
        save_weights_only=True,
        mode='max',
        save_freq=200)
    
    model=create_model_full(categorical_vars= categorical_vars,
                            transformer_vars=transformer_vars)
    export_path = os.path.join(r"C:\Users\usuario\Documents\contract-transparency-copia\python3_10\modules\models\tensor_board\keras_export")
    model.compile(optimizer=Adam(learning_rate=argumentos.learning_rate,
                                 decay=argumentos.decay),
                                 loss='mean_absolute_error',
                                 metrics=["KLDivergence","MeanSquaredError"])
    log_dir=r"C:\Users\usuario\Documents\contract-transparency - copia\python3_10\modules\models\tensor_board"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=export_path, histogram_freq=1)
    if load:
        model.load_weights(checkpointpath)
    print(categorical_vars)
    inputvar=list()
    for i in categorical_vars:
        inputvar.append(categorical_vars[i])
    inputvar.append(transformer_vars)
    out=tf.stack(output)
    if fit:
        model.fit(x=inputvar,
                  y=output,batch_size=8,epochs=1,validation_split=0.2,verbose=2,
                  callbacks=[model_checkpoint_callback,tensorboard_callback])

    tf.keras.models.save_model(model, export_path+"modelfull_tr.hdf5" )
    return model,inputvar

def full_train_categ():
    ...

#this will be later transplanted to the main file, but for now I need to test
#the code somewhere and tdidn´t find a better place
#some paths to where things are
path_to_models=r"trainedmodels"
pathToData = r"data/sucio/CONTRATOS_COVID(20-22).csv"
path_to_result=r"data\resultados"
#this cant be change, they are how rn where train (will be changed manually)
subsetsize=500000


mean=163740447
ssd=1716771980
#loading the cleaned dataset
data_pred=cleaning.secop_for_prediction(pathToData=pathToData)

#entrenar los modelos o continuar su entrenamiento
#we select from "TR" for trasformer "RN" for recurrent "NN" for neural network
#
entrenar="TR" 
setsize=2000000   
data_desc=cleaning.secop2_general(pathToData =pathToData,subsetsize=setsize)
data_categ2=cleaning.secop2_categoric(pathToData =pathToData,subsetsize=setsize)
data_categ2=data_categ2.astype(str).applymap(lambda x:[x.replace(" ","")])

data_categ=pd.DataFrame()
#generamos un tokenizer por cada categoria y lo aplicamos, mantener en falso
if False:
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
    
    data_value=cleaning.secop2_valor(pathToData =pathToData,subsetsize=setsize).apply(
        lambda x:(x-mean)/ssd)
    
    
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
data= pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = setsize/2,
      decimal = ".")

data["predict"]=predict
data["real"]=data_value["VALOR NORM".lower()]
data.to_excel(path_to_result+r"\results3.xlsx")
datatesting=data.dropna(subset=["real"])

r2_score(datatesting["predict"],datatesting["real"])

model.plot_model()
