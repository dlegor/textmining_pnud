import pandas as pd 
import numpy as np 
import dask 

from typing import Union,List,Any

import spacy
sp_nlp=spacy.load('es_core_news_md')
tokens_aux=sp_nlp('algo',disable=['entity','ner'])
nan_sentence=sp_nlp('Sin Comentario')

vector_aux=tokens_aux.vector

def vector_representation(str_vec:str)->np.ndarray:
    tokens=sp_nlp(str_vec,disable=['entity','ner'])

    if len(tokens.vector)==0:
        return vector_aux
    else:
        return tokens.vector

def vector_courpus(frame=Union[pd.DataFrame,pd.Series,List[str]])->Any:

    if isinstance(frame,pd.DataFrame):
        assert frame.shape[1]==1,"The Shape must be (n_rows,1)"
        frame_int=frame.to_list()
    
    if isinstance(frame,pd.Series):
        frame_int=frame.to_list()
    
    if isinstance(frame, list):
        frame_int=frame
    
    L=[dask.delayed(vector_representation)(x) for x in frame_int]
    L_out=dask.compute(*L)
    return np.asarray(L_out)


def _similarity_pair(sentence1:str,sentence2:str)->float:
    
    if len(sentence1.strip())<3:
        sentence1='Sin Comentario'
    
    if len(sentence2.strip())<3:
        sentence2='Sin Comentario'
    
    s1=sp_nlp(sentence1)
    s2=sp_nlp(sentence2)

    if s1.vector.size==0:
        if s2.vector.size==0:
            return 1.0
        else:
            s2.similarity(nan_sentence)
    else:
        return s1.similarity(s2)






