# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:30:02 2023

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
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from sklearn.model_selection import train_test_split
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


#entrenar los modelos o continuar su entrenamiento
#we select from "TR" for trasformer "RN" for recurrent "NN" for neural network
#
entrenar="TR" 
setsize=70000
data_desc=cleaning.secop2_general(pathToData =pathToData,subsetsize=setsize)
data_categ2=cleaning.secop2_categoric(pathToData =pathToData,subsetsize=setsize)
data_categ2=data_categ2.astype(str).applymap(lambda x:[x.replace(" ","")])
data_value=cleaning.secop2_valor(pathToData =pathToData,subsetsize=setsize).apply(
        lambda x:(x-mean)/ssd)


data_categ=pd.DataFrame()
for column in data_categ2:
    try:

        pathto_token=r"trainedmodels\tokenizers"+"\\"+column+"tokenizer.json"
        f = open(pathto_token)
        tokenizer2=tokenizer_from_json(json.load(f))
        data_categ[column]=data_categ2[column].apply(lambda x:int(tokenizer2.texts_to_sequences(x)[0][0]))
    except:
        ...

data_categ_train, data_categ_test, data_value_train, data_value_test = train_test_split(
     data_categ, data_value, test_size=0.2, random_state=153)


if True:
    reg=GradientBoostingRegressor(random_state=153,learning_rate=0.1,
                                  n_estimators=500,verbose=2,loss="squared_error",
                                  warm_start=True)
    
    hist=reg.fit(data_categ_train,data_value_train)
    


pickle.dump(reg, open('model.pkl','wb'))
predict=reg.predict(data_categ_test)
data_value_test["predict"]=predict
predict=reg.predict(data_categ)


data_value["predict"]=predict

data_categ2["predicted_valu"]=data_value["predict"].apply(lambda x:(x*ssd)+mean)
data_categ2["real_valu"]=data_value["valor norm"].apply(lambda x:(x*ssd)+mean)
data_categ2.to_excel(path_to_result+r"\results3.xlsx")


r2_score(data_value_test["valor norm"],data_value_test["predict"])


