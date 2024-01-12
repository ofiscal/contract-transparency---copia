# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:42:57 2023

@author: usuario
"""
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification,AutoTokenizer
from transformers.onnx import config
import pandas as pd
import python3_10.modules.cleaning as cleaning
import numpy as np
from tensorflow import keras
#from sklearn.metrics import r2_score
from tensorflow import keras

tf.test.is_built_with_cuda()
tf.config.run_functions_eagerly(True)
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

entrenar="TR" 
setsize=20000
data_desc=cleaning.secop2_general(pathToData =pathToData,subsetsize=setsize)

data_value=cleaning.secop2_valor(pathToData =pathToData,subsetsize=setsize).apply(
        lambda x:(x-mean)/ssd)

checkpoint="dccuchile/albert-tiny-spanish"
tokenizer=AutoTokenizer.from_pretrained(checkpoint,from_pt=True)
pre_token=data_desc["detalle del objeto a contratar"].tolist()
tokens=tokenizer(pre_token,padding=True,return_tensors="np")
tokenized_data = dict(tokens)
labels = np.array(data_value["valor norm"])
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=1,from_pt=True)

adamizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,

)

model.compile(optimizer=adamizer)

model.layers[0].trainable = False

model.fit(x=tokenized_data,y=labels,batch_size=32, epochs=6, validation_split=0.2, use_multiprocessing=True,workers=6)
results=model.predict(tokenized_data).logits
resultados=pd.DataFrame(results)
resultados.to_excel(path_to_result+r"\resultshugging.xlsx")
model.save_pretrained("/model_saved")
#r2_score(data_value["valor norm"],resultados[0])
