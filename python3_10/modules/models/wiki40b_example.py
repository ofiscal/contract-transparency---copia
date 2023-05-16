# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:09:56 2023

@author: Daniel Duque
"""
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text
tf.disable_eager_execution()

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
import tensorflow_hub as hub
import tensorflow_hub as text
import pickle
import swifter
import pandas as pd
import dask.dataframe as dd
import multiprocessing as mp
from multiprocessing import get_context

n_layer = 12
d_model = 768
max_gen_len = 1

def generate(module, inputs, mems):
    """Generate text."""
    inputs = tf.dtypes.cast(inputs, tf.int64)
    generation_input_dict = dict(input_tokens=inputs)
    mems_dict = {}
    for i in range(n_layer):
        mems_dict["mem_{}".format(i)] = mems[i]
    generation_input_dict.update(mems_dict)
      
    generation_outputs = module(generation_input_dict, signature="prediction",
                                as_dict=True)
    probs = generation_outputs["probs"]
      
    new_mems = []
    for i in range(n_layer):
        new_mems.append(generation_outputs["new_mem_{}".format(i)])

    return probs, new_mems

def generate_embedding(text1):
    
    
    
    import tensorflow.compat.v1 as tf
    import tensorflow_hub as hub
    import pandas as pd

    texto=[text1]
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    import numpy as np
    import tensorflow.compat.v1 as tf
    import tensorflow_hub as hub
    import tensorflow_text as tf_text
    tf.disable_eager_execution()

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
    import tensorflow_hub as hub
    import tensorflow_hub as text
    import pickle
    import swifter
    import pandas as pd
    import dask.dataframe as dd
    import multiprocessing as mp
    from multiprocessing import get_context
    print(1)
    g = tf.Graph()
    with g.as_default():
        module = hub.Module("https://tfhub.dev/google/wiki40b-lm-es/1")
    
        activations = module(dict(text=texto),signature="activations", as_dict=True)
        activations = activations["activations"]
        print(2)
        init_op = tf.group([tf.global_variables_initializer(),
                            tf.tables_initializer()])
        print(3)
    # Initialize session.
    with tf.Session(graph=g) as session:
        session.run(init_op)
        activations = session.run([
            activations]) 
        print(4)
    return pd.Series(tf.get_static_value(activations[0][:,:,-1].flatten()))


a=generate_embedding("cada dia vuelvo a reconocer que hay")     
      

path_to_models=r"trainedmodels"
pathToData = r"data/sucio/CONTRATOS_COVID(20-22).csv"
path_to_result=r"data/limpio"
#this cant be change, they are how rn where train (will be changed manually)

mean=163740447
ssd=1716771980
data_pred=cleaning.secop_for_prediction(pathToData=pathToData)

#entrenar los modelos o continuar su entrenamiento
#we select from "TR" for trasformer "RN" for recurrent "NN" for neural network
#
entrenar="TR" 
setsize=5
data_desc = cleaning.secop2_general(pathToData =pathToData,subsetsize=setsize)


data_desc2=data_desc["detalle del objeto a contratar"].parallel_apply(generate_embedding)

data_desc2.to_csv(path_to_result+r"\tensor_embedding.csv",decimal = ".",sep=";")



