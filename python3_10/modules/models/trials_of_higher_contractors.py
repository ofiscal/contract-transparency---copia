# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:18:41 2023

@author: Daniel Duque
"""

import pandas as pd
import datetime
import math
datos=pd.read_csv(r"C:\Users\usuario\Downloads\SECOP_II_-_Procesos_de_Contrataci_n.csv",usecols=["Precio Base",
                                                                                                 "Entidad","Nit Entidad","CodigoProveedor",
                                                                                                 "Nombre del Proveedor Adjudicado","Fecha de Publicacion del Proceso"]
                  , decimal=".")

datos["fecha"]=datos["Fecha de Publicacion del Proceso"].apply(lambda x: datetime.datetime.strptime(x,"%m/%d/%Y") if type(x)==str else datetime.datetime.strptime("01/01/1950","%m/%d/%Y"))
datos2=datos[datos["fecha"]>datetime.datetime.strptime("07/07/2022","%m/%d/%Y")]


datos["Precio Basenum"]=datos["Precio Base"].apply(lambda x: int(x.replace(",","")) if (int(x.replace(",",""))<3e12) else 0)
grouped=datos.groupby(["CodigoProveedor"]).sum().reset_index()
grouped["contratos"]=datos.groupby(["CodigoProveedor"]).count().reset_index()["Entidad"]
nombres=datos[["CodigoProveedor","Nombre del Proveedor Adjudicado"]].drop_duplicates()
grouped2=grouped.merge(nombres,how="left",on=["CodigoProveedor"])
sorteeado=grouped2.sort_values("Precio Basenum",ascending=False)

sorteeado.to_excel(r"C:\Users\usuario\Documents\contract-transparency-copia\data\limpio\contratos_ordenados2023.xlsx")
##excluir universidades públicas