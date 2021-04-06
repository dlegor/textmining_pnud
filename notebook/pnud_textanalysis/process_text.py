import re 
import dask 
import spacy 
import string
import pandas as pd
from typing import Any, Dict,List,Union,Tuple
from spacy.lang.es import STOP_WORDS



nlp_sp=spacy.load('es_core_news_md',disable=['entity','ner'])


def preprocess_text(text):
    """
    TODO:Documentantion
     1. lower case
        2. removes digits
        3. removes variations of dashes and hyphens
        Parameter
        ---------
        text: string
        Returns
        -------
        string
    """
    # lowercase and remove digits
    
    text = re.sub(r'[1-9]+[to]{2,}','#',text.lower())
    text = re.sub(r'([0-9]?)+[,.]?[0-9]+', '#', text)
    text = re.sub(r'([0-9]?)+[,.]?[0-9]+%', '%', text)
    text = re.sub(r'[# ]+%',' %',text)
    text = re.sub(r'[$]+[#]+','$',text)
    text = re.sub(r'[\r+\n?]+','',text)
    text = re.sub(r"([a-zA-Z]+)[,;:-]+",r"\1 ",text)# add space 
    text = re.sub(r"[^\w.,:#%$-]+", " ", text) # remove strange chars
    text = re.sub(r'[\s]+', r" ", text) # remove doble spaces
    text = re.sub(r'[#]{2,}|(#[- ]#)','#',text)
    text = re.sub(r'#:','',text)
    text = re.sub(r'[" "]+', r" ", text) # remove doble spaces
    text = text.strip()#remove spaces on the sides
    return text

def process_text(text:str)->str:
    """
    TODO:Docuementation
    """
    #TODO:Documentantion
    #text=re.sub(r'[#%$]+',' ',text)
    text=re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',text)
    text=re.sub(r'[" "]+',' ',text)
    return text

def get_sentences(str_inp:str)->List[str]:
    """
    TODO:Docuementation
    """

    text=preprocess_text(text=str_inp)
    doc_text=nlp_sp(text)
    return [str(s) for s in doc_text.sents]

def remove_stopwords(str_inp:str)->str:
    """
    TODO:Docuementation
    """

    return ' '.join([m for m in str_inp.split(' ') if not m in STOP_WORDS])

def clean_lemma(str_inp:str)->str:
    """
    TODO:Docuementation
    """

    doc_text=nlp_sp(str_inp)
    return ' '.join([token.lemma_ for token in doc_text])

def parallel_clean_process_basic(frame_string:Union[List[str],pd.Series])->Any:
    """
    TODO:Docuementation
    """
     
    
    func=lambda x: process_text(preprocess_text(x))

    if isinstance(frame_string,pd.Series):
        L=frame_string.tolist()
    else:
        L=frame_string
    M=[dask.delayed(func)(x) for x in L]
    return dask.compute(*M)

def parallel_clean_process(frame_string:Union[List[str],pd.Series])->Any:
    """
    TODO:Docuementation
    """
     
    
    func=lambda x: clean_lemma(remove_stopwords(process_text(preprocess_text(x))))

    if isinstance(frame_string,pd.Series):
        L=frame_string.tolist()
    else:
        L=frame_string
    M=[dask.delayed(func)(x) for x in L]
    return dask.compute(*M)

def parallel_get_sentences(frame_string:Union[List[str],pd.Series])->Any:
    """
    TODO:Docuementation
    """
     
    if isinstance(frame_string,pd.Series):
        L=frame_string.tolist()
    else:
        L=frame_string
    M=[dask.delayed(get_sentences)(x) for x in L]
    return dask.compute(*M)


__all__=['preprocess_text',
         'process_text',
         'get_sentences',
         'parallel_clean_process_basic',
         'parallel_clean_process',
         'parallel_get_sentences']