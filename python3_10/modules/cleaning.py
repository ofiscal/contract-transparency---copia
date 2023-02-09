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
      usecols = ["Descripcion del Proceso"])

    return data



def secop2_valor(subsetsize = subsetsize, # Indices into (subsetting) the data.
        pathToData=pathToData        
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
               x if x < 4e10 else 4e10 ) )
    return data


def secop2_categoric(subsetsize = subsetsize, # Indices into (subsetting) the data.
        pathToData=pathToData        
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
      usecols = names2,
      decimal = ".",)


    return data



def secop2_numeric(subsetsize = subsetsize, # Indices into (subsetting) the data.
        pathToData=pathToData        
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