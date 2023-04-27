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
setsize=40000
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


reg=GradientBoostingRegressor(random_state=0,learning_rate=0.01,n_estimators=100000)

hist=reg.fit(data_categ,data_value)

predict=reg.predict(data_categ)
data_value["predict"]=predict



data_value.to_excel(path_to_result+r"\results3.xlsx")


r2_score(data_value["valor norm"],data_value["predict"])