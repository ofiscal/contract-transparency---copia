# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:51:08 2024

@author: usuario
"""

from transformers import pipeline
import pandas as pd
import torch
import sys
# Load pre-trained model
nlp = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",device=0)







def is_sustainable(text):
    # Define sustainability-related keywords
    keywords = ["agua","energia","biodiversidad","cambio climático",
                "residuos sólidos","calidad atmosférica","calidad del aire",
                "generación de empleo","costo total de propiedad","otros"]

    try:
        result = nlp(text,candidate_labels=keywords)["labels"]
    except KeyboardInterrupt:
        sys.exit()
    except:
        result="nada"
    print("procesado")
    return result


# Test the function
text = "estamos procurando la reduccion de gases con efecto invernadero"
is_sustainable(text)


casa=pd.read_csv(r"C:\Users\usuario\Documents\contract-transparency-copia\data\sucio\SECOP_II_-_Contratos_Electr_nicos.csv",skiprows=0,nrows=190000)

device = "cuda:0" if torch.cuda.is_available() else "cpu"



casa["label sostenible"]=casa["Objeto del Contrato"].apply(is_sustainable)

casa.to_csv(r"C:\Users\usuario\Documents\contract-transparency-copia\data\resultados\sostenibles.csv")    
"""
"""