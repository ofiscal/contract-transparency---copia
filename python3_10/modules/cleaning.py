# Cleaning for SECOP 2 dataset and a test dataset.

from typing import Tuple
import pandas as pd


# TODO : Should the following two things
# be defined in open code (i.e. importable into another module)
# the way they are now,
# or only defined in the function below that uses them?
pathToData = r"data/SECOP_II_-_Contratos_Electr_nicos.csv"
subsetsize=10000

def secop2_general (
    subsetsize = subsetsize, # Indices into (subsetting) the data.
    pathToData = pathToData
    ) ->  pd.DataFrame:

  # TODO : This should be rewritten as:
  # def clean ( skiprows : pd.Series) -> pd.DataFrame:
  #   df = pd.read_csv (
  #     ...
  #     skiprows = skiprows,
  #     ... )
  #   df["Valor del Contrato"] = ...
  #   return df
  # test = clean( skiprows = subset[0] )
  # datatest = clean( skiprows = ... )

    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      decimal = ".",
      usecols = ["Descripcion del Proceso","Valor del Contrato"])

    data ["Valor del Contrato"] = (
      data["Valor del Contrato"]
      . apply ( lambda x:
                float(x.replace(",","")) )
      . apply( lambda x:
               x if x < 4e10 else 4e10 ) )
    return data

def secop2_categorical(
        pathToData=pathToData        
        ) ->pd.DataFrame:
    ...
    names=[]

def secop_for_prediction(
        pathToData=pathToData        
        ) ->pd.DataFrame:
    #load the predicted data
    data = pd.read_csv (
      pathToData,
      skiprows =[1,subsetsize],
      nrows = subsetsize,
      decimal = ".",)
    data ["Valor del Contrato"] = (
      data["Valor del Contrato"]
      . apply ( lambda x:
                float(x.replace(",","")) )
      . apply( lambda x:
               x if x < 4e10 else 4e10 ) )
    
    
    return data