# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:21:09 2024

@author: usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:55:27 2023

@author: Daniel Duque
"""



import pandas as pd


from sklearn.metrics import r2_score
import json

import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#this will be later transplanted to the main file, but for now I need to test
#the code somewhere and tdidn´t find a better place
#some paths to where things are
path_to_models=r"trainedmodels"
pathToData = r"C:\Users\usuario\Documents\contract-transparency-copia\data\resultados\torch_weights2.xlsx"
path_to_general=r"C:\Users\usuario\Documents\contract-transparency-copia\data\limpio\value_col.csv"

#this cant be change, they are how rn where train (will be changed manually)



#takes only the regresive variables 
data_categ=pd.read_excel(pathToData).take([i for i in range(1,1501)],axis=1)
data_categ2=pd.read_csv(path_to_general)

data_value=pd.read_csv(pathToData)["valor_real"]



data_categ_train, data_categ_test, data_value_train, data_value_test = train_test_split(
     data_categ, data_value, test_size=0.15, random_state=1,shuffle=False)

dtrain_reg = xgb.DMatrix(data_categ_train, data_value_train)
dtest_reg = xgb.DMatrix(data_categ_test, data_value_test)
pure_data = xgb.DMatrix(data_categ, data_value)


if True:
    n = 5000
    params = {"objective": "reg:squarederror","reg_alpha":1000,"reg_lambda":1000}
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


