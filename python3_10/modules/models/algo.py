# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:21:45 2023

@author: usuario
"""
# librerías (por orden alfabético)


import numpy as np
import pandas as pd
import os


from sklearn.ensemble import GradientBoostingRegressor

path_file = r"C:\Users\usuario\Downloads\Absenteeism_at_work.csv"
path = r"C:\Users\usuario\Downloads\Absenteeism_at_work.csv"
def load_clean(path=path_file):
        # funcion para leer el archivo en formato csv 
    
    df = pd.read_csv(path,sep=";")
    print(df)
    columns=df.columns.tolist()
    columns.remove("Absenteeism time in hours")
    print(columns)
    return df,columns

def multivariate (path=path_file,load=True):
    if load:

        df_abs,columns = load_clean (path)
    reg = GradientBoostingRegressor()

    reg.fit(df_abs[columns],df_abs["Absenteeism time in hours"])
    return reg
df,columns=load_clean()
mean=df["Absenteeism time in hours"].mean()
model = multivariate(path=path_file)
model.get_params()
predict=pd.DataFrame(model.predict(df[columns]))
df["importancia"]=predict[0].apply(lambda x: "importante" if x>=mean else "moderado")
df["predict"]=predict[0]
print(df)