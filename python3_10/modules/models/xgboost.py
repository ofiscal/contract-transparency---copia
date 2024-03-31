# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:55:27 2023

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
import xgboost as xgb
from sklearn.metrics import mean_squared_error
#this will be later transplanted to the main file, but for now I need to test
#the code somewhere and tdidn´t find a better place
#some paths to where things are
path_to_models=r"trainedmodels"
pathToData = r"data/sucio/CONTRATOS_COVID(20-22).csv"
path_to_result=r"data/resultados"
hugging=path_to_result+r"\resultshugging.xlsx"
#this cant be change, they are how rn where train (will be changed manually)



mean=163740447
ssd=1716771980
#loading the cleaned dataset


#entrenar los modelos o continuar su entrenamiento
#we select from "TR" for trasformer "RN" for recurrent "NN" for neural network
#
entrenar="TR" 
setsize=2000

data_desc=cleaning.secop2_general(pathToData =pathToData,subsetsize=setsize)
data_categ2=cleaning.secop2_categoric(pathToData =pathToData,subsetsize=setsize)
data_categ2=data_categ2.astype(str).applymap(lambda x:[x.replace(" ","")])
data_hugging=pd.read_excel(hugging)[0]
#data_categ2["hugging"]=data_hugging
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
data_categ["hugging"]=data_hugging
data_categ_train, data_categ_test, data_value_train, data_value_test = train_test_split(
     data_categ, data_value, test_size=0.2, random_state=153)

dtrain_reg = xgb.DMatrix(data_categ_train, data_value_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(data_categ_test, data_value_test, enable_categorical=True)
pure_data = xgb.DMatrix(data_categ, data_value, enable_categorical=True)


if True:
    n = 5000
    params = {"objective": "reg:squarederror","reg_alpha":0.2,"reg_lambda":0.2}
    evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    reg = xgb.train(
       params=params, 
       dtrain=dtrain_reg,
       num_boost_round=n,
       evals=evals,
       verbose_eval=25
       )
    

pickle.dump(reg, open('modelxgboost.pkl','wb'))

predict=reg.predict(pure_data)


data_value["predict"]=predict

data_categ2["predicted_valu"]=data_value["predict"].apply(lambda x:(x*ssd)+mean)
data_categ2["real_valu"]=data_value["valor norm"].apply(lambda x:(x*ssd)+mean)

def flagging(predict,real):
    if predict<=0:
        predict=1000000
    if (real-predict)>0:
        size=(real-predict)/(real+1)
        value=size*(real-predict)
    else:
        size=0
        value=0
    return float(size),float(value)

data_categ2[["probability","value in danger"]]=data_categ2.apply(lambda row:
    flagging(row["predicted_valu"],row["real_valu"]),axis=1, result_type="expand")
    
data_full = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = setsize,
      decimal = ".",)

data_full[["probability","value in danger"]]=data_categ2.apply(lambda row:
    flagging(row["predicted_valu"],row["real_valu"]),axis=1, result_type="expand")
data_full["predicted_valu"]=data_categ2["predicted_valu"]
data_full["real_valu"]=data_categ2["real_valu"]
data_full.to_csv(path_to_result+r"\results3.csv",decimal = ".",sep=";")



r2_score(data_full["real_valu"],data_full["predicted_valu"])


