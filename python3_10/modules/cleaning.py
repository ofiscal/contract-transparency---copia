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

def secop2_general (
    subsetsize = subsetsize, # Indices into (subsetting) the data.
    pathToData = pathToData
    ) ->  pd.DataFrame:



    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      decimal = ".",
      usecols = ["Descripcion del Proceso"])

    return data



def secop2_valor(subsetsize:int = subsetsize, # Indices into (subsetting) the data.
        pathToData: str | os.PathLike =pathToData       
        ) ->pd.DataFrame:
    ...
    


    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      decimal = ".",
      usecols = ["Valor del Contrato"])

    data ["Valor del Contrato"] = (
      data["Valor del Contrato"]
      . apply ( lambda x:
                float(x.replace(",","")) )
      . apply( lambda x:
               x if x < 1e10 else 1e10 ) )
    return data


def secop2_categoric(subsetsize = subsetsize, # Indices into (subsetting) the data.
        pathToData: str | os.PathLike=pathToData        
        ) ->pd.DataFrame:
    ...
    names=['Nombre Entidad', 'Departamento', 'Ciudad',
           'Orden', 'Sector', 'Rama', 'Entidad Centralizada',
           'Estado Contrato','Tipo de Contrato',
           'Modalidad de Contratacion', 'Justificacion Modalidad de Contratacion',
           'Condiciones de Entrega',
           'TipoDocProveedor', 'Proveedor Adjudicado',
           'Es Grupo', 'Es Pyme', 'Habilita Pago Adelantado', 'Liquidación',
           'Obligación Ambiental', 'Obligaciones Postconsumo', 'Reversion',
           'Estado BPIN', 'Código BPIN', 'Anno BPIN','EsPostConflicto', 'Destino Gasto',
           'Origen de los Recursos', 'Puntos del Acuerdo',
           'Pilares del Acuerdo', 'Nombre Representante Legal',
           'Nacionalidad Representante Legal',
           'Tipo de Identificación Representante Legal',
           'Identificación Representante Legal', 'Género Representante Legal',#ojo
           ]

    names2=['Es Grupo', 'Es Pyme','TipoDocProveedor','Nombre Entidad',
            'Departamento','Ciudad','Estado Contrato',
            'Tipo de Contrato','Nacionalidad Representante Legal',
            'Origen de los Recursos', 
            'Género Representante Legal', 'Rama']
    """

    """
    data = pd.read_csv (
      pathToData,
      skiprows =0,
      nrows = subsetsize,
      usecols = names,
      decimal = ".",)


    return data

def secop2_date(subsetsize = subsetsize, # Indices into (subsetting) the data.
        pathToData: str | os.PathLike=pathToData        
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

    return data

def secop2_numeric(subsetsize = subsetsize, # Indices into (subsetting) the data.
        pathToData: str | os.PathLike=pathToData        
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
    return data



def secop_for_prediction(
        pathToData: str | os.PathLike=pathToData        
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








