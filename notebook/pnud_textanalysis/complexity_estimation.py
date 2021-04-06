from pandas.io import feather_format
from .complexity import Long_Sentences
from .complexity import Long_Words
from .complexity import Complex_Content
from .complexity import LexicalRichness
from .complexity import Meta_Data_Text
from .complexity import preprocess_text

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import spacy
import pandas as pd
import numpy as np
from typing import Callable,Any

class Complexity_Index:
    """
    TODO:Documentation
    """
    def __init__(self,nlp_spacy:Any,
                preprocessor:Callable=preprocess_text,
                contamination:float=0.01,
                n_components:int=10,
                limit:float=0.85) -> None:

        self.nlp_spacy=nlp_spacy
        self.preprocessor=preprocessor
        self.contamination=contamination
        self.n_components=n_components
        self.limit=limit
    

    def fit(self,X:pd.Series):
        """
        TODO:Documentation
        """

        X=X.fillna('Sin documentación sobre la Causa.')
        
        Step_1=X.apply(lambda x:Meta_Data_Text(text_input=x,nlp_spacy=self.nlp_spacy).get_all_metrics())
        Step_2=X.apply(lambda x:Long_Words(text_input=x,nlp_spacy=self.nlp_spacy).get_all_metrics())
        Step_3=X.apply(lambda x:Long_Sentences(text_input=x,nlp_spacy=self.nlp_spacy).get_all_metrics())
        Step_4=X.apply(lambda x:Complex_Content(text_input=x,nlp_spacy=self.nlp_spacy).get_all_metrics())
        Step_5=X.apply(lambda x:LexicalRichness(text_input=x,preprocessor=self.preprocessor).get_all_metrics())

        Output=pd.DataFrame()

        Output.loc[:,'len_str']=Step_1.apply(lambda x:x[0])
        Output.loc[:,'num_sentence']=Step_1.apply(lambda x:x[2])
        Output.loc[:,'mean_len_by_words']=Step_1.apply(lambda x:x[4])
        Output.loc[:,'ration_words']=Step_1.apply(lambda x:x[6])
        Output.loc[:,'mean_chars_pers_words']=Step_2.apply(lambda x:x[0])
        Output.loc[:,'mean_syllabels_per_words']=Step_2.apply(lambda x:x[3])
        Output.loc[:,'words_at_least_3_syllables']=Step_2.apply(lambda x:x[4])
        Output.loc[:,'words_with_fewer_3_syllables']=Step_2.apply(lambda x:x[5])
        Output.loc[:,'words_with_1_syllables']=Step_2.apply(lambda x:x[7])
        Output.loc[:,'mean_chars_per_sentences']=Step_3.apply(lambda x:x[0])
        Output.loc[:,'mean_senteces_len_syllables']=Step_3.apply(lambda x:x[2])
        Output.loc[:,'proportion_nouns']=Step_4.apply(lambda x:x[0])
        Output.loc[:,'proportion_verd']=Step_4.apply(lambda x:x[1])
        Output.loc[:,'proportion_adjetives']=Step_4.apply(lambda x:x[2])
        Output.loc[:,'proportion_adverbs']=Step_4.apply(lambda x:x[3])
        Output.loc[:,'nwords']=Step_4.apply(lambda x:x[4])
        Output.loc[:,'ttr']=Step_5.apply(lambda x: x[0])
        Output.loc[:,'rttr']=Step_5.apply(lambda x: x[1])
        Output.loc[:,'msttr']=Step_5.apply(lambda x: x[7])
        Output.loc[:,'mtld']=Step_5.apply(lambda x: x[8])
        
        #Null
        Output.fillna(0,inplace=True)
        
        #Normalization and Outliers
        outlieres = [('minmax_trans', MinMaxScaler()), 
        ('lof', LocalOutlierFactor(n_jobs=-1,contamination=self.contamination))]
        pipe1=Pipeline(outlieres)
        V1=pipe1.fit_predict(Output)

        Output=Output.loc[V1==1,:]

        #Normalization and PCA
        pca_index = [('transformation', StandardScaler()), 
        ('pca', PCA(n_components=self.n_components))]
        pipe2=Pipeline(pca_index)
        P=pipe2.fit_transform(Output)
       
        #Explained Variance
        r=pipe2.named_steps['pca'].explained_variance_ratio_.cumsum()
        m=self.find_index_var(r,limit=self.limit)
        
        #Index
        CI=P[:,:(m+1)].mean(axis=1)
        
        #Transformation of the Index
        #CII=MinMaxScaler(feature_range=(-1,1)).fit_transform(CI.reshape(-1,1))
        CII=PowerTransformer().fit_transform(CI.reshape(-1,1))
        
        CIII=MinMaxScaler(feature_range=(-1,1)).fit_transform(CII)
        
        #Index Segmentation
        A=pd.qcut(CIII.squeeze(),5,labels=False)
        DOut=pd.DataFrame(index=Output.index,data={'Indicador_Complejidad':CIII.squeeze(),'Segmentos_Indicador':A})

        return DOut
    
    @staticmethod
    def find_index_var(arr:np.ndarray,limit:float):
        n=len(arr)
        i=0
        m=0
        while i<n:
            if arr[i]<=limit and arr[i]>=0.8:
                m=max(m,i)
                i+=1
            else:
                i+=1
        return m

