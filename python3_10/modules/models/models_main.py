# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:18:20 2023

@author: Daniel Duque Lozano
"""

import tensorflow as tf
from tensorflow import keras
import python_3_10.modules.models_main.Transformers as Transformers
import pandas as pd

class joining():
    def __init__(self):
        ...
    
    def load_models(self,path_to_models):
        self.Transformer=Transformers.create_model_tr()
        self.Transformer.load_weights(path_to_models+"\model1_tr.hdf5")
        
    
    def train_model (self,
        model:str,#Transformer, RNN, or...
        epocas:float,
        X: pd.Series,
        Y:pd.Series,
        path_to_models
        ):
        
        if model=="Transformer":
            self.Transformer=Transformers.keep_train(self.Transformer,
                        epocas=10,
                        X=X,
                        Y=Y,
                        checkpointpath=path_to_models+"\model1_tr.hdf5")
        
    
    def __add__(self,
        models=list
                ):
        ...
    
        