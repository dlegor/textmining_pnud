from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List,Union
import pandas as pd
import numpy as np

MODELS=["roberta-base-nli-stsb-mean-tokens","stsb-roberta-large","xlm-r-bert-base-nli-stsb-mean-tokens"]

class Create_Embeddings:
    """
    TODO:Documentation
    """
    def __init__(self,name_model:str="roberta-base-nli-stsb-mean-tokens",
                device:str=None,
                multiple_sentences:bool=False) -> None:
        self.name_model=name_model
        self.device=device
        self.multiple_sentences=multiple_sentences

        #Load the Model
        if self.name_model in MODELS:
            model = SentenceTransformer(self.name_model,device=self.device)
            self.embding_dim=model.get_sentence_embedding_dimension()
            self.model=model
        else:
            #TODO:Description
            raise ValueError("Error")

    def fit(self,X:Union[List[str],pd.Series])->None:
        """
        Documentation
        """
        #Num Rows 
        n=len(X)

        if isinstance(X,pd.Series):

            if self.multiple_sentences:
                embedding_matrix = np.zeros((n,self.embding_dim))
                for i,txt in enumerate(X.tolist()):
                    embedding_matrix[i]=np.mean(self.model.encode(txt.tolist()),axis=0)
                
                return embedding_matrix
            else:
                embedding_matrix = np.zeros((n,self.embding_dim))
                for i,txt in enumerate(X.tolist()):
                    embedding_matrix[i]=self.model.encode(str(txt).replace('\n',''))

                return embedding_matrix
        
        if isinstance(X,list):

            if self.multiple_sentences:

                embedding_matrix = np.zeros((n,self.embding_dim))
                for i,txt in enumerate(X):
                    embedding_matrix[i]=np.mean(self.model.encode(txt.tolist()),axis=0)
                
                return embedding_matrix
                
            else:
                embedding_matrix = np.zeros((n,self.embding_dim))
                for i,txt in enumerate(X):
                    embedding_matrix[i]=self.model.encode(str(txt).replace('\n',''))

                return embedding_matrix






        



