from .sillabizer import Syllabizer
import spacy
import numpy as np

class Long_Words(Syllabizer):
    """
    Extraction of meters at word level in each sentence
    """
    
    def __init__(self, text_input:str,nlp_spacy:spacy.lang.es.Spanish):
        
        self.text_input=text_input
        self._doc_spacy=nlp_spacy(text_input)
        
    @staticmethod
    def _is_word(doc_inp):
        """Return ``True`` if ``pos_tag`` is not punctuation, False otherwise.
        This method checks that the ``pos_tag`` does not belong to the following
        set: ``{'PUNCT', 'SYM', 'SPACE'}``.
        :param pos_tag: Part of Speech tag
        :type pos_tag: string
        :return: True if POS is a word
        :rtype: boolean
        """
        if (doc_inp.pos_ != 'PUNCT' and doc_inp.pos_ != 'SYM' \
            and doc_inp.pos_ != 'SPACE' and doc_inp.pos_ != 'NUM'):
            return True
        else:
            return False

    @property
    def mean_chars_pers_words(self)->float:
        return np.mean([len(x) for x in self._doc_spacy if self._is_word(x)])
            
    @property
    def words_at_least_7_chars(self)->float:
        return sum([1 if len(x)>=7 else 0 \
                              for x in self._doc_spacy if self._is_word(x)])
    @property
    def words_at_least_6_chars(self)->float:
        return sum([1 if len(x)>=6 else 0 \
                                    for x in self._doc_spacy if self._is_word(x)])
        
    @property
    def mean_syllabels_per_words(self)->float:
        
        return np.mean([self.number_of_syllables(s.text) \
                                          for s in self._doc_spacy if self._is_word(s)])
    
    @property
    def words_at_least_3_syllables(self)->float:
        return sum([1 if self.number_of_syllables(s.text)>=3 else 0 \
                                        for s in self._doc_spacy if self._is_word(s)])
    
    @property
    def words_with_fewer_3_syllables(self)->float:
        return sum([1 if self.number_of_syllables(s.text)<3 else 0 \
                                          for s in self._doc_spacy if self._is_word(s)])
    @property
    def words_with_fewer_3_syllables(self)->float:
        return sum([1 if self.number_of_syllables(s.text)<3 else 0 \
                                          for s in self._doc_spacy if self._is_word(s)])
    @property
    def words_with_2_syllables(self)->float:
        return sum([1 if self.number_of_syllables(s.text)==2 else 0 \
                                    for s in self._doc_spacy if self._is_word(s)])
    @property
    def words_with_1_syllables(self)->float:
        return sum([1 if self.number_of_syllables(s.text)==1 else 0 \
                                    for s in self._doc_spacy if self._is_word(s)])
    
    def get_all_metrics(self):      
        return (round(self.mean_chars_pers_words,3),round(self.words_at_least_7_chars,3),
                round(self.words_at_least_6_chars,3),round(self.mean_syllabels_per_words,3),
                round(self.words_at_least_3_syllables,3),round(self.words_with_fewer_3_syllables,3),
                round(self.words_with_2_syllables,3),round(self.words_with_1_syllables,3))
        

class Long_Sentences(Syllabizer):
    """
    Extraction of metrics at sentence level
    """
    
    def __init__(self, text_input:str,nlp_spacy:spacy.lang.es.Spanish):

        self.text_input=text_input
        self._doc_spacy=nlp_spacy(text_input)

                
    @staticmethod
    def _is_word(doc_inp):
        """Return ``True`` if ``pos_tag`` is not punctuation, False otherwise.
        This method checks that the ``pos_tag`` does not belong to the following
        set: ``{'PUNCT', 'SYM', 'SPACE'}``.
        :param pos_tag: Part of Speech tag
        :type pos_tag: string
        :return: True if POS is a word
        :rtype: boolean
        """
        if (doc_inp.pos_ != 'PUNCT' and doc_inp.pos_ != 'SYM' \
            and doc_inp.pos_ != 'SPACE' and doc_inp.pos_ != 'NUM'):
            return True
        else:
            return False
    
    @property
    def mean_chars_per_sentences(self):
        return np.mean([len(s) for x in self._doc_spacy.sents for s in x])
    
    @property
    def mean_setences_len_words(self):
        
        mean_len_words=np.mean([len(w) for w in self._doc_spacy.sents])
        return mean_len_words

    @property
    def mean_senteces_len_syllables(self):

        msls=np.mean([self.number_of_syllables(m.text) for w in self._doc_spacy.sents \
                                             for m in w if self._is_word(m)])
        return msls
        
    def get_all_metrics(self):
               
        return (round(self.mean_chars_per_sentences,3),round(self.mean_setences_len_words,3),
                round(self.mean_senteces_len_syllables,3))
        
class Complex_Content(Syllabizer):
    """
    Metrics extraction at the sentence structure level (part of speech)
    """
    
    def __init__(self, text_input:str,nlp_spacy:spacy.lang.es.Spanish):
        self.text_input=text_input
        self._doc_spacy=nlp_spacy(text_input)

    
        
    @staticmethod
    def is_word(doc_inp):
        """Return ``True`` if ``pos_tag`` is not punctuation, False otherwise.
        This method checks that the ``pos_tag`` does not belong to the following
        set: ``{'PUNCT', 'SYM', 'SPACE'}``.
        :param pos_tag: Part of Speech tag
        :type pos_tag: string
        :return: True if POS is a word
        :rtype: boolean
        """
        if (doc_inp.pos_ != 'PUNCT' and doc_inp.pos_ != 'SYM' \
            and doc_inp.pos_ != 'SPACE' and doc_inp.pos_ != 'NUM'):
            return True
        else:
            return False
        
    @property
    def proportion_nouns(self):
        d=sum([1 for s in self._doc_spacy if self.is_word(s)])
        n=sum([1  for x in self._doc_spacy if x.pos_=='NOUN'])
        return n/d if d!=0 else 0
    @property
    def proportion_verbs(self):
        d=sum([1 for s in self._doc_spacy if self.is_word(s)])
        n=sum([1  for x in self._doc_spacy if x.pos_=='VERB'])
        return n/d if d!=0 else 0
    @property
    def proportion_adjetives(self):
        d=sum([1 for s in self._doc_spacy if self.is_word(s)])
        n=sum([1  for x in self._doc_spacy if x.pos_=='ADJ'])
        return n/d if d!=0 else 0
    @property
    def proportion_adverbs(self):
        d=sum([1 for s in self._doc_spacy if self.is_word(s)])
        n=sum([1  for x in self._doc_spacy if x.pos_=='ADV'])
        return n/d if d!=0 else 0
    @property
    def no_words(self):
        nw=sum([1 for s in self._doc_spacy if not self.is_word(s)])
        return nw 


    def get_all_metrics(self):
        proportion_nouns=self.proportion_nouns
        proportion_verd=self.proportion_verbs
        proportion_adjetives=self.proportion_adjetives
        proportion_adverbs=self.proportion_adverbs
        nwords= self.no_words
        
        return (round(proportion_nouns,3),round(proportion_verd,3),
                round(proportion_adjetives,3),round(proportion_adverbs,3),round(nwords,3))
