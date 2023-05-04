# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:52:56 2023

@author: usuario
"""
#this is based on word embeddings from https://github.com/dccuchile/spanish-word-embeddings
import io
import re
import string
import tqdm

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from gensim.models.wrappers import FastText
wordvectors_file_vec = r'python3_10\modules\pre-processing\embeddings-xs-model.vec'

wordvectors = FastText.load_fasttext_format(wordvectors_file_vec)
hi=["hola","como","estas"]
vector = wordvectors.vocab