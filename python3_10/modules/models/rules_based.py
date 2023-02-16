
#this module provides a series of functions that gives indications for
#how extrange a contract may be.
import pandas as pd
import numpy as np
import datetime



def date_near_change(date:datetime,
                     neardates:list(),
                     tipo_de_contrat:str())->float:

    #this function makes a weirdness indicator for certain contracts as they are more near to changes in government.
    #dependiendo del tipo de contrato audiencias de centros poblados... modos de selección, mini licitación
    #ley de garantias 4 meses antes, contratación directa no, justo antes de la vijencia disparan 
    #tipo de contratos que son usados antes de la ley de garantias para desviar, contratación directa , convenios interadministrativos, 
    #convenios de asociación "vagabunderia".
    diff=neardates.map(lambda x: x-datetime)
    mindiff=diff.min()
    return mindiff.day()/365
def date_near_control(date,neardates):
    #this function makes a weird indicator for contrats that are near to "control de garantias".
    ...
    
def licitación_un_oferente(numero_de_oferentes):
    #indica el numero de oferentes
    return 

def fechas_atípicos_de_duración_licitación_contrato(modelo_de_predicción )
    #si las fechas son diferentes 

def otrosies()    
    #adición al contrato inicial, cuando nos parezca sospechoso... ojos contratos
    #obra pública, se adicionan(un cuarto del valor inicial)
    #merge por bpin, si tiene el mismo codigo bpin puede tener varios,
    #regex adición, otrosí, otro si, otrosi.
    
def contratación_directa_prestación(columna="codigo de proveedor"
                                    persona_natural=True, en_ejecución=True):
    #tiene que ser persona natural
    regla=datos[columna].apply(lambda x: 0 if x<4 else x)
    return regla

def big_contracts(estimarlo):
    #ver porcentaje del gasto del municipio en un solo contrato según el tipo de persona
    
def proveedor_monopolistico():#chulito :D
    #numero de empresas con el mismo representante
    
def cne_cuentas(registro_de_donantes, representante_legal,cedula):
    #TODO




def ponderador():
    #a builder that acumulates all the information in the previous functions in this modules and build
    #a single indicator for extrangenes.
    #
    
    ...
    
    
    
def test_near_change():
    neardates=[datetime.datetime.strptime("10/11/2022","%d/%m/%Y"),datetime.datetime.strptime("10/9/2022","%d/%m/%Y")]
    
    
