# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:16:43 2023

@author: Daniel Duque
"""

import pandas as pd
#this module is dedicated to distance functions.

def distance_percentage(
        X1:pd.Series,
        X2:pd.Series
        )->pd.Series:
    #calculates the percentage distance of a series
    return (X1-X2)/X2

