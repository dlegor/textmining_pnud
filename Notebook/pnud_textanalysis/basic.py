import pandas as pd 
import numpy as np 
import re 
import dask 
from collections import Counter
from typing import Any,List
from spacy.lang.es import STOP_WORDS
import os 
import seaborn as sns
import spacy

cm = sns.light_palette("green", as_cmap=True)

sp_nlp=spacy.load('es_core_news_sm')

# Cols=['JUSTIFICACION_AJUSTE_ENERO_MAY','AVANCE_CAUSA_MES1','AVANCE_CAUSA_MES2','AVANCE_CAUSA_MES3','AVANCE_CAUSA_MES4','AVANCE_CAUSA_MES5',
#       'AVANCE_CAUSA_MES6','AVANCE_CAUSA_MES7','AVANCE_CAUSA_MES8','AVANCE_CAUSA_MES9','AVANCE_CAUSA_MES10','AVANCE_CAUSA_MES11',
#       'AVANCE_CAUSA_MES12','AVANCE_ENERO_MAYO_CAUSA','AVANCE_EFECTO_MES1','AVANCE_EFECTO_MES2','AVANCE_EFECTO_MES3',
#       'AVANCE_EFECTO_MES4','AVANCE_EFECTO_MES5','AVANCE_EFECTO_MES6','AVANCE_EFECTO_MES7','AVANCE_EFECTO_MES8','AVANCE_EFECTO_MES9',
#       'AVANCE_EFECTO_MES10','AVANCE_EFECTO_MES11','AVANCE_EFECTO_MES12','AVANCE_ENERO_MAYO_EFECTO','AVANCE_OTROS_MOTIVOS_MES1',
#       'AVANCE_OTROS_MOTIVOS_MES2','AVANCE_OTROS_MOTIVOS_MES3','AVANCE_OTROS_MOTIVOS_MES4','AVANCE_OTROS_MOTIVOS_MES5','AVANCE_OTROS_MOTIVOS_MES6',
#       'AVANCE_OTROS_MOTIVOS_MES7','AVANCE_OTROS_MOTIVOS_MES8','AVANCE_OTROS_MOTIVOS_MES9','AVANCE_OTROS_MOTIVOS_MES10','AVANCE_OTROS_MOTIVOS_MES11',
#       'AVANCE_OTROS_MOTIVOS_MES12','AVANCE_ENERO_MAYO_OTROS_MOTIVO','AVANCE_CAUSA_CP','AVANCE_EFECTO_CP','AVANCE_OTROS_MOTIVOS_CP']



def load_data(file_path:str,columns:List[str]):
    if not os.path.isfile(file_path):
        raise ValueError(" The file path is wrong")
    if os.path.splitext(file_path)[1]!='.csv':
        raise ValueError("The extension file is wrong")
        
    try:
        data=pd.read_csv(filepath_or_buffer=file_path,
                         encoding='latin-1',
                         low_memory=False)
    except IOError as ioe:
        raise (ioe)

    if len(columns)==0 :return data
    else: return data[columns]


def report_missinvalues(frame:pd.DataFrame):
    return (frame.isnull().sum()/frame.shape[0])\
           .to_frame(name='% Missin Values').style.background_gradient(cmap=cm)

def len_string(s):
    return len(str(s))

def len_approx_words(s):
    return len(str(s).split(' '))


def report_leght_string(frame:pd.DataFrame)->pd.DataFrame:
    DF=frame.applymap(len_string)
    return DF.where(DF!=3,0).describe().style.background_gradient(cmap=cm)

def report_num_words(frame:pd.DataFrame)->pd.DataFrame:
    DF=frame.applymap(len_approx_words)
    return DF.where(DF!=1,0).describe().style.background_gradient(cmap=cm)

def plot_string(frame:pd.DataFrame,type_plot:str)->Any:
    if type_plot=='len_string':
        DF=frame.applymap(len_string)
        (DF[DF>3].mean()).plot(kind='bar',title='Length per Character')
    elif type_plot=='len_words':
        DF=frame.applymap(len_approx_words)
        (DF[DF>1].mean()).plot(kind='bar',title='Length per Word')
    else:
        raise ValueError("Type_plot unknown")


def aggregation_columns(frame:pd.DataFrame,name_col:str,name_serie:str)->pd.Series:
    Cols=frame.columns.tolist()
    COLUMNS=[s for s in Cols if s.startswith(name_col)]
    Serie_Out=frame[COLUMNS].astype(str).agg(' '.join,axis=1)
    Serie_Out.name=name_serie
    Serie_Out=Serie_Out.apply(lambda x: x.replace('nan','').strip())
    return Serie_Out


def Freq_Words(SWords:List[str])->Any:
    List_words=[]
    for wd in SWords:
        List_words+=wd.split(" ")
    
    Counter_1=Counter(List_words)
    return Counter_1

def plot_freq_words_df(most_occur:Any,row:int)->Any:
    Row=row
    p1=pd.DataFrame(most_occur,columns=['Words','Frequency'])\
         .assign(Frequency_Normaliazation=lambda X:X['Frequency'].div(Row))\
         .filter(items=['Frequency_Normaliazation','Words'])\
         .set_index('Words')\
         .plot(kind='barh')
    return p1

def clean_without_stop(SWords:List[str])->Any:
    List_words=[]
    for wd in SWords:
        L=[s for s in wd.split(" ") if (s not in STOP_WORDS) and (s!='')]
        List_words+=L
    
    Counter_1=Counter(List_words)
    return Counter_1

def cleaning_senteces(sentences:str)->str:
    doc_text=sp_nlp(sentences)
    return ' '.join([token.lemma_ for token in doc_text if (token.is_punct==False and token.is_stop==False)])

def clean_lemma(SWords:List[str],nrow:int=10000)->Any:
    L=[dask.delayed(cleaning_senteces)(x) for x in SWords[:nrow]]
    return dask.compute(*L)


