import spacy 
import string
from typing import Any
from numpy import mean as np_mean

class Meta_Data_Text(object):
    """
    Class to get the basic look of the string text.
    """
    def __init__(self,text_input:str,
                 nlp_spacy:Any,
                 punctua:str=string.punctuation):
        
        self.str_inp=str(text_input)
        self.doc_=nlp_spacy(text_input)
        self.punct=punctua
        
    @property
    def len_str(self)->int:
        """
        string length
        """
        return len(self.str_inp)

    @property
    def num_blocks(self)->int:
        """
        string splitted by blocks
        """
        return len(self.str_inp.split(" "))

    @property
    def num_sentence(self)->int:
        """ 
        number of sentences
        """
        return len([d for d in self.doc_.sents])

    @property
    def num_words_without_stopword(self)->int:
        """ 
        Get words that are not stop words
        """
        return len([d for d in self.doc_ if d.is_stop==False and \
                d.is_punct==False])

    @property
    def mean_len_by_words(self)->float:
        """
        mean number of words
        Example:
        >>> m='Sin Documentación del indicador'
        >>> mean_len_bywords(m)
        >>> 7
        """
        str_inp_=self.str_inp
        return round(np_mean(list(map(len,str_inp_.split(" ")))),2)

    @property
    def num_punctuation(self)->int:
        """
        number of punctuation symbols
        """
        return len([s for s in self.str_inp if s in self.punct])
    
    def ration_words(self):
        a=self.num_words_without_stopword
        b=self.num_blocks
        if b==0:
            return 0
        else:
            return round(a/b,2)
        
    def get_all_metrics(self):
        """
        Function to get all metrics
        """
        return (self.len_str,self.num_blocks,self.num_sentence,
                self.num_words_without_stopword,self.mean_len_by_words,self.num_punctuation,self.ration_words())