#!/usr/bin/env python
# coding: utf-8

# # Limpieza de la Base de Datos
# 
# Este script contiene todas las funciones necesarias para limpiar y transformar la base en la versión más adecuada para el proyecto.
# 
# Los objetivos son los siguientes:
#     
#     * Depurar los campos que contengan texto con algún error de escritura o caracteres inadecuados para el procesamiento.
#     * Fijar un tipo de dato adecuado para el manejo con otros archivos
#     * Compactar u organizar las columnas que en un formato que permita una mejor visualización y organización de la información referente a los reportes mensuales.
# 

#Required packages

import pandas as pd
import numpy as np
import warnings

import typer
from wasabi import msg
from pathlib import Path
from typing import Union

#warning messages are ignored
warnings.filterwarnings("ignore")


# Constants for the cleaning process
SCHEMA={'CICLO':'category',
 'ID_RAMO':'category',
 'DESC_RAMO':'category',
 'ID_UR':'category',
 'DESC_UR':'category',
 'ID_ENTIDAD_FEDERATIVA':'category',
 'ENTIDAD_FEDERATIVA':'category',
 'ID_MUNICIPIO':'category',
 'MUNICIPIO':'category',
 'GPO_FUNCIONAL':'category',
 'DESC_GPO_FUNCIONAL':'category',
 'ID_FUNCION':'category',
 'DESC_FUNCION':'category',
 'ID_SUBFUNCION':'category',
 'DESC_SUBFUNCION':'category',
 'ID_AI':'category',
 'DESC_AI':'category',
 'ID_MODALIDAD':'category',
 'DESC_MODALIDAD':'category',
 'ID_PP':'category',
 'DESC_PP':'category',
 'MODALIDAD_PP':'category',
 'ID_PND':'category',
 'DESC_PND':'category',
 'OBJETIVO_PND':'category',
 'PROGRAMA_PND':'category',
 'DESC_PROGRAMA_PND':'category',
 'OBJETIVO_PROGRAMA_PND':'category',
 'DESC_OBJETIVO_PROGRAMA_PND':'category',
 'OBJETIVO_ESTRATEGICO':'category',
 'ID_NIVEL':'category',
 'DESC_NIVEL':'category',
 'INDICADOR_PND':'category',
 'TIPO_RELATIVO':'category',
 'FRECUENCIA':'category',
 'TIPO_INDICADOR':'category',
 'DIMENSION':'category',
 'UNIDAD_MEDIDA':'category',
 'SENTIDO':'category'
 }

DTIPO_INDICADOR={'Estratégico':'Estratégico',
 'Gestión':'Gestión',
 'Sesiones de Comité Técnico':'Sesiones de Comité Técnico',
 'Gestion':'Gestión',
 'SOLICITUDES DE SERVICIO':'Solicitudes de Servicio',
 'ECONOMIA':'Economía',
 'Estrategico':'Estratégico',
 'gestión':'Gestión',
 'Absoluto':'Absoluto',
 'Sectorial':'Sectorial',
 'Desempeño Operativo':'Desempeño Operativo',
 'GESTION':'Gestión',
 'ESTRATÉGICO':'Estratégico',
 'De Gestión':'Gestión',
 'Estratgico':'Estratégico'}

DDIMENSION={'Eficacia':'Eficacia',
 'Eficiencia':'Eficacia',
 'Economía':'Economía',
 'Calidad':'Calidad',
 'eficacia':'Eficacia',
 'ECONOMIA':'Economía',
 '0':'Sin Dato',
 'Servicios Personales':'Servicios Personales',
 'Económica':'Economía',
 'Eificacia':'Eficacia',
 'EFICACIA':'Eficacia',
 'Eficiciencia':'Eficiencia',

 'Es la suma ponderada de la proporción de las observaciones de alto impacto respecto del total de observaciones determinadas en las auditorías directas de alto impacto realizadas por el área de Auditoría Interna del OIC; la calidad de dichas observaciones, y la calidad de las recomendaciones que de éstas se derivan. (Eficacia)':'Eficacia',

 'Es un promedio ponderado que evalúa al OIC en la atención de quejas y denuncias. (Eficacia)':'Eficacia',

 'Mide las acciones de las Áreas de Responsabilidades en algunas de sus funciones primordiales: 1) el tiempo en la atención de los expedientes, 2) la resolución de expedientes y 3) la firmeza de las sanciones impuestas. (Eficacia)':'Eficacia',

 'PORCENTAJE DE SOLICITUDES DE PRÉSTAMO AUTORIZADAS':'Porcentaje de Solicitudes de Préstamo Autorizadas',

 'El Indicador de Mejora de la Gestión (IMG) evalúa las acciones realizadas por los OIC en sus instituciones de adscripción y en aquellas bajo su atención, así como los resultados alcanzados en las mismas. Específicamente, el indicador se orienta a evaluar la manera en que los OIC:\r\n\r\n- Promueven acciones orientadas al logro de resultados respecto a las vertientes comprometidas en sus Programas Anuales de Trabajo (PAT)2015, en materia de auditoría para el desarrollo y mejora de la gestión pública.':'Sin Datos'}


# Auxiliary functions

def cln_txt(str_inp:str)->str:
    """
    remove special characters
    """

    str_inp=str_inp.replace(u'\xa0',u' ')
    str_inp=str_inp.replace(u'\n',u' ')
    str_inp=str_inp.replace(u'\r',u' ')
    txt=''.join([s for s in str_inp if not s in '!"#$%&\'()*+-;<=>?@[\\]^_`{|}~' ])
    return txt.replace('  ','').strip()

def main(path_file:str=typer.Argument(...,help='Path to input file')):
    """
    Limpieza de la Base de Datos
    
    Este script contiene todas las funciones necesarias para limpiar y transformar la base en la versión más adecuada para el proyecto.

    Los objetivos son los siguientes:
     
     * Depurar los campos que contengan texto con algún error de escritura o caracteres inadecuados para el procesamiento.
     * Fijar un tipo de dato adecuado para el manejo con otros archivos
     * Compactar u organizar las columnas que en un formato que permita una mejor visualización y organización de la información referente a los reportes mensuales.

    """

    #Load Data
    msg.info("Load data...")
    data=pd.read_csv(path_file,encoding='latin1',low_memory=False,dtype=SCHEMA)

    msg.info("General Information:\n")
    msg.info(data.info())
    #Remove rows with NIVEL== FID
    msg.good("Remove FID...")
    data=data[data.DESC_NIVEL!='FID'].copy()

    #Fields cleaning DESCRIPCIONES(DES_*)
    msg.info("Cleaning DESC_ ...")
    data.DESC_RAMO=data.DESC_RAMO.apply(lambda x: cln_txt(str(x)))
    data.DESC_UR=data.DESC_UR.apply(lambda x: cln_txt(str(x)))
    data.DESC_AI=data.DESC_AI.apply(lambda x: cln_txt(str(x)))
    data.DESC_PP=data.DESC_PP.apply(lambda x: cln_txt(str(x)))
    data.OBJETIVO_PND=data.OBJETIVO_PND.apply(lambda x: cln_txt(str(x)))
    data.DESC_OBJETIVO_PROGRAMA_PND=data.DESC_OBJETIVO_PROGRAMA_PND.apply(lambda x: cln_txt(str(x)))
    data.OBJETIVO_ESTRATEGICO=data.OBJETIVO_ESTRATEGICO.apply(lambda x: cln_txt(str(x)))
    data.DESC_MATRIZ=data.DESC_MATRIZ.apply(lambda x: cln_txt(str(x)))
    data.DESC_OBJETIVO=data.DESC_OBJETIVO.apply(lambda x: cln_txt(str(x)))

    #Change wrong names
    msg.info("Changes names...")
    data.TIPO_INDICADOR=data.TIPO_INDICADOR.map(DTIPO_INDICADOR)
    data.DIMENSION=data.DIMENSION.map(DDIMENSION)

    #Change data type
    msg.info("Change data type...")
    data.ID_OBJETIVO=data.ID_OBJETIVO.astype('int')
    data.ID_OBJETIVO_PADRE=data.ID_OBJETIVO_PADRE.fillna(-1).astype('int')
    data.ID_INDICADOR_CICLO_ANTERIOR=data.ID_INDICADOR_CICLO_ANTERIOR.fillna(-1).astype('int')
    data.CICLO_LINEA_BASE=data.CICLO_LINEA_BASE.fillna(-1).astype('int')

    #List of columns to group data
    msg.info("Create List of Columns...")
    META_MES_COL=data.columns[data.columns.str.startswith('META_MES')].tolist()
    META_AJUSTADA_MES_COL=data.columns[data.columns.str.startswith('META_AJUSTADA_MES')].tolist()
    AVANCE_MES_COL=data.columns[data.columns.str.startswith('AVANCE_MES')].tolist()
    JUSTIFICACION_AJUSTE_MES_COL=data.columns[data.columns.str.startswith('JUSTIFICACION_AJUSTE_MES')].tolist()
    AVANCE_CAUSA_MES_COL=data.columns[data.columns.str.startswith('AVANCE_CAUSA_MES')].tolist()
    AVANCE_EFECTO_MES_COL=data.columns[data.columns.str.startswith('AVANCE_EFECTO_MES')].tolist()
    AVANCE_OTROS_MOTIVOS_MES_COL=data.columns[data.columns.str.startswith('AVANCE_OTROS_MOTIVOS_MES')].tolist()

    #META by months
    msg.info("Meta by months...")
    for i in range(12):
        data[f'RECORDS_META_MES{i+1}']=(data[f'META_MES{i+1}'].astype('string')+':'\
            +data[f'META_MES{i+1}_NUM'].astype('string')+':'+data[f'META_MES{i+1}_DEN']\
                .astype('string'))

    #META AJUSTADA by months
    msg.info("Meta Ajustada by months...")
    for i in range(12):
        data[f'RECORDS_META_AJUSTADA_MES{i+1}']=(data[f'META_MES{i+1}'].astype('string')\
            +':'+data[f'META_MES{i+1}_NUM'].astype('string')+':'+data[f'META_MES{i+1}_DEN']\
                .astype('string'))

    #AVANCE by months
    msg.info("AVANCE by months...")
    for i in range(12):
        data[f'RECORDS_AVANCE_MES{i+1}']=(data[f'META_MES{i+1}'].astype('string')+':'+\
            data[f'META_MES{i+1}_NUM'].astype('string')+':'+data[f'META_MES{i+1}_DEN']\
                .astype('string'))

    #JUSTIFICACION by Months
    msg.info("JUSTIFICACION by months...")

    func='|'.join

    data['JUSTIFICACIONES_AJUSTE_POR_MES']=data[JUSTIFICACION_AJUSTE_MES_COL]\
        .fillna('#').astype('str').apply(lambda x:func(x),axis=1)

    #AVANCE CAUSA by months
    msg.info("AVANCE CAUSA by months...")
    data['AVANCE_CAUSA_POR_MES']=data[AVANCE_CAUSA_MES_COL].fillna('#').astype('str')\
    .apply(lambda x:func(x),axis=1)

    #AVANCE EFECTO by Months
    msg.info("AVANCE EFECTO by months...")
    data['AVANCE_EFECTO_POR_MES']=data[AVANCE_EFECTO_MES_COL].fillna('#').astype('str')\
    .apply(lambda x:func(x),axis=1)

    #AVANCE OTROS MOTIVOS by months
    msg.info("AVANCE OTROS MOTIVOs by months...")
    data['AVANCE_OTROS_MOTIVOS_POR_MES']=data[AVANCE_OTROS_MOTIVOS_MES_COL].fillna('#')\
    .astype('str').apply(lambda x:func(x),axis=1)

    #Delete columns group data
    msg.info("delete columns")
    data.drop(labels=META_MES_COL+META_AJUSTADA_MES_COL+AVANCE_MES_COL,inplace=True,axis=1)
    data.drop(labels=JUSTIFICACION_AJUSTE_MES_COL+AVANCE_CAUSA_MES_COL+
            AVANCE_EFECTO_MES_COL+AVANCE_OTROS_MOTIVOS_MES_COL,inplace=True,axis=1)

    msg.info("General Information:\n")
    data.info()


    #Save File
    msg.info("Save the Files...")
    data.reset_index().to_feather('base')#Para version feather
    #data.to_csv('base.csv.zip',encoding='latin1', index=False,compression='zip')# Para guardad en versión csv
    msg.good("OK!!!")


if __name__=='__main__':
    typer.run(main)