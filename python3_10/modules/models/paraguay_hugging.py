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

path_to_data=r"C:\Users\usuario\Documents\contract-transparency-copia\data\sucio\records.csv"
#This parameters are used to normalize, however they may be misleading as they
#Aren´t really means or standard deviation in any actual structure
mean=0
ssd=1e+17


entrenar="TR"
setsize=20000
#we run a subset of the 2023 dataset from contracts in paraguay open data
#we can find the source here https://www.contrataciones.gov.py/datos/api/v3/doc/
#As we need complet information to train we drop empty values and normalize
data=pd.read_csv(path_to_data,nrows=setsize)
data=data.dropna(subset="compiledRelease/tender/value/amount")
data=data.dropna(subset="compiledRelease/planning/budget/description")
data["valor norm"]=data["compiledRelease/tender/value/amount"].apply(
    lambda x:(x-mean)/ssd
)





checkpoint="dccuchile/albert-tiny-spanish"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
pre_token=data["compiledRelease/planning/budget/description"].tolist()
tokens=tokenizer(pre_token,padding=True,return_tensors="np")
tokenized_data = dict(tokens)
labels = np.array(data["valor norm"])
model = TFAutoModelForSequenceClassification.from_pretrained(
    checkpoint,num_labels=1,from_pt=True)

adamizer=tf.keras.optimizers.Adam(
    learning_rate=1e-12,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-03,
    #clipvalue=1e-03

)

model.compile(optimizer=adamizer)

model.layers[0].trainable = False

model.fit(x=tokenized_data,y=labels,batch_size=64, epochs=1, validation_split=0.2,use_multiprocessing=True,workers=6)
results=model.predict(tokenized_data)
resultados=pd.DataFrame(results.logits)
resultados2=resultados[0].apply(lambda x: (x*ssd)+mean)
data=data.reset_index()
data["Precio Predecido"]=resultados2
data.to_csv(r"data/resultados/resultshugging.csv", sep=";")

model.save_pretrained(r"model_saved")
tokenizer.save_pretrained(r"model_saved")
#r2_score(data_value["valor norm"],resultados[0])
