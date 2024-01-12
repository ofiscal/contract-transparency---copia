# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:09:59 2023

@author: Daniel Duque
"""

import pandas as pd
from pyspark.sql import SparkSession
#Create PySpark SparkSession


data=pd.read_csv("data/sucio/SECOP_II_-_Contratos_Electr_nicos.csv",nrows=10000)
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()
sparkDF=spark.createDataFrame(data) 
esquema=sparkDF.schema.json()