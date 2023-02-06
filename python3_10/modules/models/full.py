# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:01:39 2023

@author: Daniel Duque
"""


def create_model_categorical(
        categorical_vars:pd.DataFrame(),
        numerical_vars:pd.DataFrame(),
        transformer_vars:pd.DataFrame(),
        conv_vars:pd.DataFrame()
        )->tf.keras.Model:
    
    