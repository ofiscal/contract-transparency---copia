# Cleaning for SECOP 2 dataset and a test dataset.

from typing import Tuple
import pandas as pd
import numpy as np
import os
# TODO : Should the following two things
# be defined in open code (i.e. importable into another module)
# the way they are now,
# or only defined in the function below that uses them?
pathToData = r"data/SECOP_II_-_Contratos_Electr_nicos.csv"
subsetsize=10000
rand_porc=1
rand_seed=24111996
def secop2_general (
    subsetsize = subsetsize, # Indices into (subsetting) the data.
    rand:int=rand_porc,
    pathToData = pathToData,
    rand_seed=rand_seed
    ) ->  pd.DataFrame:



    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      decimal = ".",
      usecols = ["Detalle del Objeto a Contratar"])
    data.columns= data.columns.str.lower()
    return data



def secop2_valor(subsetsize:int = subsetsize, # Indices into (subsetting) the data.
                 rand:int=rand_porc,
        pathToData: str =pathToData,
        rand_seed=rand_seed
        ) ->pd.DataFrame:
    ...
    


    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      decimal = ".",
      usecols = ["VALOR NORM"])
    try:
        data ["VALOR NORM"] = (
          data["VALOR NORM"]
          . apply ( lambda x:
                    float(x.replace(",","").replace("$","")))
          . apply( lambda x:
                   x if x < 1e100 else 1e10 ) )
    except:
        data ["VALOR NORM"] = (
          data["VALOR CONTRATO NORMA"]
          . apply ( lambda x:
                    float(x.replace(",","").replace("$","")))
          . apply( lambda x:
                   x if x < 1e100 else 1e10 ) )
        
    data.columns= data.columns.str.lower()
    return data


def secop2_categoric(subsetsize = subsetsize, # Indices into (subsetting) the data.
                     rand:int=rand_porc,
        pathToData: str =pathToData,
        rand_seed=rand_seed        
        ) ->pd.DataFrame:
    ...
    names=[ 'Anno Cargue SECOP','Anno Firma del Contrato',
           'Nivel Entidad','Orden Entidad',"MPPIO ENTIDAD NORM","Condiciones de Entrega","ORIGEN RECURSOS NORM",
           "Nombre Grupo","DEPTO ENTIDAD NORM",
           'TIPO PROCESO NORM','Estado del Proceso',
           'ID Regimen de Contratacion','Tipo de Contrato',
           "CAUSALES CONT DIRECTA NORM","Plazo de Ejec del Contrato"
           ]

    
    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      usecols = names,
      decimal = ".",)
    data.columns= data.columns.str.lower()

    return data

def secop2_date(subsetsize = subsetsize, # Indices into (subsetting) the data.
                rand:int=rand_porc,
        pathToData: str =pathToData,
        rand_seed=rand_seed        
        ) ->pd.DataFrame:
    ...
    names=[ 'Fecha de Firma', 'Fecha de Inicio del Contrato',
             'Fecha de Fin del Contrato', 'Fecha de Inicio de Ejecucion',
             'Fecha de Fin de Ejecucion'
           ]


    """

    """
    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      usecols = names,
      decimal = ".",)
    data=pd.to_datetime(data)
    data.columns= data.columns.str.lower()
    return data

def secop2_numeric(subsetsize = subsetsize, # Indices into (subsetting) the data.
                   rand:int=rand_porc,
        pathToData: str =pathToData,
        rand_seed=rand_seed        
        ) ->pd.DataFrame:
    ...
    names=['Codigo Proveedor'
           ]


    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      usecols=names,
      decimal = ".",)

    data['Codigo Proveedor']  = data['Codigo Proveedor']. apply ( lambda x:
                float(x.replace(",",""))) 
    data.columns= data.columns.str.lower()
    return data



def secop_for_prediction(
        pathToData: str =pathToData        
        ) ->pd.DataFrame:
    #load the predicted data
    data = pd.read_csv (
      pathToData,
      skiprows =[1,subsetsize],
      nrows = subsetsize,
      decimal = ".",)

        
    try:
        data ["VALOR NORM"] = (
          data["VALOR NORM"]
          . apply ( lambda x:
                    float(x.replace(",","").replace("$","")))
          . apply( lambda x:
                   x if x < 1e100 else 1e10 ) )
    except:
        data ["VALOR NORM"] = (
          data["VALOR CONTRATO NORMA"]
          . apply ( lambda x:
                    float(x.replace(",","").replace("$","")))
          . apply( lambda x:
                   x if x < 1e100 else 1e10 ) )
            
    data.columns= data.columns.str.lower()    
    
    return data

def new_data(
        identi:str,#key to assert new contratcs
                X:pd.DataFrame,#series old
                y:pd.DataFrame#series new
                )->pd.Series:
        #this function should receive two series and identify the new values in
        #the series, then return the new row in the new data set to load the 
        #new contract... or other things

        new= y.merge(X,on=identi, how="outer",indicator=True,
                       suffixes=('', '_y')
                       ).query('_merge=="left_only"')
        new.drop(new.filter(regex='_y$').columns, axis=1, inplace=True)
        return new
        
def new_data_test(
        ):
    X = pd.DataFrame(
    {
        "col1": ["a", "a", "b", "b", "a"],
        "col2": [1.0, 2.0, 3.0, np.nan, 5.0],
        "col3": [1.0, 2.0, 3.0, 4.0, 5.0]
    },
    columns=["col1", "col2", "col3"],
    )
    y = X.copy()
    y.loc[0, 'col1'] = 'c'
    y.loc[2, 'col3'] = 4.0
    y
    print(new_data(identi="col1",X=X,y=y))   








