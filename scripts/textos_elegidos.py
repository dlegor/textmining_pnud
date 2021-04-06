#!/usr/bin/env python
# coding: utf-8

# ## Textos Elegibles para los Experimentos
# 
# En el siguiente código selecciona aquellos indicadores que pueden ser objeto de estudio, esto considerando los siguentes criterios:
#  * Cuenta con información reportada en AVANCE_CP
#  * Cuenta con información reportada en META_AJUSTADA_CP
#  * Cuenta con información reportada en SENTIDO
#  
#  Se hace una estimación del valor que podría tener el avance, aunque existen algunos casos de TIPO_RELATIVO reportadas de menera diferente, la gran mayoría se presentan como una división entre el (AVANCE_CP / META_AJUSTADA_CP)*100.
#  
#  Sobre de este cálculo se dividen los indicadores en 3 categorias:
#  * ALTO: Indicador que reporta más del 105% de avance
#  * BAJO: Indicador que reporta menos del 95% de avance
#  * CORRECTO: Indicador que reporta un avance mayor a 95% y menos al 105%.
#  
#  Para todos los indicadores que son identificados con avance **BAJO** son divididos en 5 categorias según el avance reportado.


#Bibliotecas mínimas requeridas
import pandas as pd
import numpy as np
import typer

from pathlib import Path
from wasabi import msg

#Funciones auxiliares
def estimation_meta(type_sense:str,reached:float,approved:float):
    if type_sense=='Ascendente':
        if approved!=0:
            return (reached/approved)*100
        else:
            return np.NaN
    if type_sense=='Descendente':
        if approved!=0:
            return (reached/approved)*100
        else: return np.NaN

def funct1(x):
    return estimation_meta(x[0],x[2],x[1])

def main(path_file:str=typer.Argument(...,help='Path to input file')):
    """
    Textos Elegibles para los Experimentos

    Para ver más información revisar el Notebook Limpieza_Base.ipynb o las notas de este Script.
    """
    #Se carga el archivo
    msg.info("Se carga el archivo")
    data=pd.read_feather(path_file)
    print(data.info())
    #Condiciones sobre AVANCE_CP y META_AJUSTADA_CP
    Data=data.loc[(data.AVANCE_CP.notna()),:]\
             .loc[(data.META_AJUSTADA_CP.notna()),:]\
             .loc[(data.SENTIDO.notna()),:]

    #Condiciones sobre los Textos
    Len_Sentence_Txt=Data.AVANCE_CAUSA_CP.apply(lambda x: len(str(x).split()) if x!=None else 0)
    Data=Data[Len_Sentence_Txt>=2].copy()

    #Creación de la variable de Progreso de la Meta
    V1=Data[['SENTIDO','META_AJUSTADA_CP','AVANCE_CP']].apply(funct1,axis=1)
    Data.loc[:,'PROGRESO_META']=V1.copy()

    # Creación de Categorías por el nivel de progreso reportado
    Data.loc[:,'INDICADORA_CUMPLIMIENTO_AVANCE']='CORRETO'
    Data.loc[Data.PROGRESO_META<95,'INDICADORA_CUMPLIMIENTO_AVANCE']='BAJO'
    Data.loc[Data.PROGRESO_META>105,'INDICADORA_CUMPLIMIENTO_AVANCE']='ALTO'
    Data.loc[Data.PROGRESO_META.isnull(),'INDICADORA_CUMPLIMIENTO_AVANCE']='ERROR'


    #Revisión del conteo por categoría
    Data['INDICADORA_CUMPLIMIENTO_AVANCE'].value_counts()


    # **Nota**: La categoria "ERROR" se genera debido a que el AVANCE_CP puede ser cero, por lo cual para los fines del proyecto no resultan      		relevantes.

    #Segmentación de los indicadores con Progreso BAJO
    Data.loc[Data.INDICADORA_CUMPLIMIENTO_AVANCE=='BAJO','SEGMENTOS_INCUMPLIMIENTOS']=pd.qcut(Data[Data.INDICADORA_CUMPLIMIENTO_AVANCE=='BAJO']['PROGRESO_META'],5,labels=False).values

    Data['SEGMENTOS_INCUMPLIMIENTOS'].fillna(-1,inplace=True)

    #Se omiten aquellos indicadores que el progreso es NaN
    Data2=Data[Data.PROGRESO_META.notna()][['index','ID_INDICADOR','PROGRESO_META','INDICADORA_CUMPLIMIENTO_AVANCE','SEGMENTOS_INCUMPLIMIENTOS']]

    #Se guarda el archivo
    path1=Path('.').resolve().parent/'data'/'base_txt_prog'
    Data2.reset_index().to_feather(path1.as_posix())#Para version feather

    Base_Textos=data[data['index'].isin(Data2['index'])][['index','ID_MATRIZ','ID_INDICADOR','AVANCE_CAUSA_CP','AVANCE_EFECTO_CP','AVANCE_OTROS_MOTIVOS_CP']].copy()

    # Se guarda el archivo
    path2=Path('.').resolve().parent/'data'/'base_txt'
    Base_Textos.reset_index().to_feather(path2.as_posix())#Para version feather
    msg.good("Se ha guardado la base")

if __name__=='__main__':
    typer.run(main)
