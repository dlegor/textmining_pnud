from nlpaug.util import action
from sentence_transformers import SentenceTransformer
from pathlib import Path
from spacy.lang.es import STOP_WORDS
from typing import List,Union
from sklearn.preprocessing import normalize
from wasabi import msg

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.distributions import ECDF

from ..process_text import get_sentences

import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
import shutil

MODELS=["roberta-base-nli-stsb-mean-tokens","stsb-roberta-large","xlm-r-bert-base-nli-stsb-mean-tokens"]


def get_aug_type_1(top_k:int=10,
                  aug_max:int=5,
                  model_path:str='distilroberta-base',
                  action:str='substitute'):
    return naw.ContextualWordEmbsAug(
        model_path=model_path, 
        action=action,top_k=top_k,
        stopwords=list(STOP_WORDS),aug_max=aug_max)

def get_aug_type_2(aug_max:int=7,
                   action:str="swap"):
    return naw.RandomWordAug(action=action,stopwords=list(STOP_WORDS),aug_max=aug_max)


class Similarity_SentencesEmbedding:
    def __init__(self,path_embedding_sentences:Union[str,Path],
                path_cuases:Union[str,Path],
                path_embedding_causes:Union[str,Path],
                by_augmentation:int=3,
                sentences_embedding_model:str="roberta-base-nli-stsb-mean-tokens"
                ):
        
        self.path_embedding_sentences=path_embedding_sentences
        self.path_embedding_causes=path_embedding_causes
        self.path_cuases=path_cuases
        self.by_augmentation=by_augmentation
        self.sentences_embedding_model=sentences_embedding_model

        #Load the Model
        if self.sentences_embedding_model in MODELS:
            model = SentenceTransformer(self.sentences_embedding_model)
            self.embding_dim=model.get_sentence_embedding_dimension()
            self.model=model
        else:
            #TODO:Description
            raise ValueError("Error")



    def fit_batch(self,iteration:int=20)->np.ndarray:
        
        if self.path_cuases!=None:
            self.cause_file=True
        
        embedding_sentences=np.load(self.path_embedding_sentences)
        embedding_causes=np.load(self.path_embedding_causes)
        
        if self.cause_file:
            DF_Causes=pd.read_feather(self.path_cuases)
        
        msg.info("Star Training")

        aug_insert=get_aug_type_1(action="insert")
        aug_substitute = get_aug_type_1()
        aug_swap = get_aug_type_2(action='swap')
        aug_remove = get_aug_type_2(action='delete')

        def get_augmented_data(str_inp:str,n_out:int=self.by_augmentation)->List[str]:
            #insert
            list_insert=[aug_insert.augment(str_inp) for i in range(n_out)]
            #substitute
            list_substitute=[aug_substitute.augment(str_inp) for i in range(n_out)]
            #swap
            list_swap=[aug_swap.augment(str_inp) for i in range(n_out)]
            #Remove
            list_remove=[aug_remove.augment(str_inp) for i in range(n_out)]
            #Almos Synonimos
            return list_insert+list_substitute+list_swap+list_remove


        ESentences=normalize(X=embedding_sentences)
        
        #Similarity
        if self.cause_file:
            self.path_temp=Path('temp_matrix')
            self.path_temp.mkdir()
            i=0
            while i <iteration:
                Lista_Causas_Examples=DF_Causes.groupby('Causa_Reportada')\
                                      .AVANCE_CAUSA_TXT_1\
                                      .sample(n=1).tolist()
                LCE=[get_augmented_data(a) for a in Lista_Causas_Examples]
                Vec=[np.mean(self.model.encode(b),axis=0) for b in LCE]
                MCausas=np.mean([np.asarray(Vec),embedding_causes],axis=0)
                AVG1_Nor=normalize(X=MCausas)
                S=(ESentences@AVG1_Nor.T)
                path_matrix=self.path_temp/f'matrix_{i}'
                np.save(file=path_matrix,arr=S)

                msg.info(f"Traing {i} Iteration!!!")
                i+=1

            S=self._aggregation_matrix()
            shutil.rmtree(self.path_temp)      
        
            #return S
        else:

            i=0
            while i <iteration:
                AVG1_Nor=normalize(X=embedding_causes)
                S=(ESentences@AVG1_Nor.T)
                path_matrix=self.path_temp/f'matrix_{i}'
                np.save(file=path_matrix,arr=S)

                msg.info(f"Traing {i} Iteration!!!")
                i+=1

            S=self._aggregation_matrix()
            shutil.rmtree(self.path_temp)
            #return S    
        
        #Levels
        Cols_Names=['Causa_1','Causa_2','Causa_3','Causa_4',\
                    'Causa_5','Causa_6','Causa_7','Causa_8','Causa_9']
        
        lm=LinearRegression(n_jobs=-1)
        Distances=pd.DataFrame(data=(1.-S),columns=Cols_Names)
        
        Output={}
        Cache={}
        for j in range(1,10):
          X1=Distances[f'Causa_{j}']
          ecdf=ECDF(X1)
          for i in np.arange(0.2,0.41,0.001):
            i=round(i,4)
            X=Distances[f'Causa_{j}'][(Distances[f'Causa_{j}']>=i) \
                                      & (Distances[f'Causa_{j}']<(i+0.5))]
            Y=ecdf(X)
            lm.fit(X=X.values.reshape(-1,1),y=np.log(1-Y).reshape(-1,1))
            Yhat=lm.predict(X.values.reshape(-1,1))
            mserror=mean_squared_error(np.log(1-Y).reshape(-1,1),Yhat)
            Cache[i]=mserror
          Output[j]=min(Cache,key=Cache.get)

        self.cut_level=Output

        #Creation of Dataframe
        L1=np.where(Distances.Causa_1<Output[1],1,0)
        L2=np.where(Distances.Causa_2<Output[2],1,0)
        L3=np.where(Distances.Causa_3<Output[3],1,0)
        L4=np.where(Distances.Causa_4<Output[4],1,0)
        L5=np.where(Distances.Causa_5<Output[5],1,0)
        L6=np.where(Distances.Causa_6<Output[6],1,0)
        L7=np.where(Distances.Causa_7<Output[7],1,0)
        L8=np.where(Distances.Causa_8<Output[8],1,0)
        L9=np.where(Distances.Causa_9<Output[9],1,0)

        S_Levels=pd.DataFrame({'Causa_1':L1,'Causa_2':L2,
                               'Causa_3':L3,'Causa_4':L4,
                               'Causa_5':L5,'Causa_6':L6,
                               'Causa_7':L7,'Causa_8':L8,
                               'Causa_9':L9})

        Table_S=pd.DataFrame(data=S,columns=Cols_Names)
        

        return Table_S,S_Levels
                

    def _aggregation_matrix(self)->np.ndarray:
        List_Files=sorted(list(self.path_temp.glob('*.npy')))
        Cache_Matrix=[]
        for file_i in List_Files:
            finput=np.load(file_i)
            Cache_Matrix.append(finput)
        return np.mean(Cache_Matrix,axis=0)

    def fit(self,x:str):
        
        if self.path_cuases!=None:
            self.cause_file=True
        
        embedding_causes=np.load(self.path_embedding_causes)
        
        if self.cause_file:
            DF_Causes=pd.read_feather(self.path_cuases)
        
        vec=get_sentences(str(x))
        vec=self.model.encode(vec) 
        ESentences=normalize(vec)
        ECauses=normalize(embedding_causes)

        S=(ESentences@ECauses.T)

        return S


    def __repr__(self) -> str:
        return f"Sentence Embedding with NLP augmentation{self.by_augmentation}"