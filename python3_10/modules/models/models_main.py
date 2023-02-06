# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:18:20 2023

@author: Daniel Duque Lozano
"""

import tensorflow as tf
from tensorflow import keras
import python_3_10.modules.models_main.Transformers as Transformers
import pandas as pd

class arguments():
    def __init__(self,vocab_size: int,
                 embedding_dim: int,
                 max_length: int,
                 num_epochs: int,
                 learning_rate: float,
                 decay: float,
                 num_heads: int,
                 ff_dim: int,
                 max_word:int):
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.max_length=max_length
        self.num_epochs=num_epochs
        self.learning_rate=learning_rate
        self.decay=decay
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.max_word=max_word
        
        
argumentos=arguments(
    vocab_size=100000,
    embedding_dim=100,
    max_length=200,
    num_epochs=1,
    learning_rate=0.001,
    decay=0.00001,
    num_heads = 2,
    ff_dim = 32,
    max_word=10000,    
    )



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
    
        