"""
This code is an adaptation of the code:
https://github.com/LSYS/LexicalRichness/blob/master/lexicalrichness/lexicalrichness.py
"""
from itertools import islice
from collections import Counter
from math import sqrt, log
from scipy.stats import hypergeom
from statistics import mean
from typing import Callable
from .utils import preprocess_text
import re
import string


def segment_generator(List, segment_size):
    """ Split a list into s segments of size r (segment_size).
        Parameters
        ----------
        List: list
            List of items to be segmented.
        segment_size: int
            Size of each segment.
        Returns
        -------
        Generator
    """
    for i in range(0, len(List), segment_size):
        yield List[i: i + segment_size]

def list_sliding_window(sequence, window_size=2):
    """ Returns a sliding window generator (of size window_size) over a sequence. Taken from
        https://docs.python.org/release/2.3.5/lib/itertools-example.html
        Example
        -------
        List = ['a', 'b', 'c', 'd']
        window_size = 2
        list_sliding_window(List, 2) -> ('a', 'b')
                                        ('b', 'c')
                                        ('c', 'd')
        Parameters
        ----------
        sequence: sequence (string, unicode, list, tuple, etc.)
            Sequence to be iterated over. window_size=1 is just a regular iterator.
        window_size: int
            Size of each window.
        Returns
        -------
        Generator
    """
    iterable = iter(sequence)
    result = tuple(islice(iterable, window_size))
    if len(result) == window_size:
        yield result
    for item in iterable:
        result = result[1:] + (item,)
        yield result

class LexicalRichness(object):
    """ Object containing tokenized text and methods to compute Lexical Richness (also known as
        Lexical Diversity or Vocabulary Diversity.)
    """

    def __init__(self, text_input:str, 
                 preprocessor:Callable=preprocess_text, 
                 tokenizer:bool=False):
        """ Initialise object with basic attributes needed to compute the common lexical diversity measures.
            Parameters
            ----------
            text: string or list
                String (or unicode) variable containing textual data, or a list
                of tokens if the text is already tokenized.
            preprocessor: callable or None
                A callable for preprocessing the text. Default is the built-in
                `preprocess` function. If None, no preprocessing is applied.
            tokenizer: callable or None
                A callable for tokenizing the text. Default is the built-in
                `tokenize` function. If None, the text parameter should be a list.
            Attributes
            ----------
            wordlist: list
                List of tokens from text.
            words: int
                Number of words in text.
            terms: int
                Number of unique terms/vocabb in text.
            preprocessor: callable
                The preprocessor used.
            tokenizer: callable
                The tokenizer used.
            Helpers Functions
            -----------------
            preprocess(string):
                Preprocess text before tokenizing (if preprocessor=preprocess)
            blobber(string)
                Tokenize text using TextBlob (if tokenizer=blobber)
            tokenize(string)
                Tokenize text using built-in string methods (if tokenizer=tokenize)
        """
        
        #self.text = preprocess(text_input)
        self.tokenizer = tokenizer
    
        

        if callable(preprocessor):
            text=preprocessor(text_input)
            if self.tokenizer:
                text=preprocessor(text_input)
                self.wordlist=self._tokenize(text)
            else:
                self.wordlist = self._process_text(text).split( )
        else:
            self.wordlist=self._tokenize(text)
            


        self.words = len(self.wordlist)
        self.terms = len(set(self.wordlist))
        
    @staticmethod
    def _process_text(text:str)->str:
        text=re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',text)
        text=re.sub(r'[" "]+',' ',text)
        return text

    @staticmethod
    def _tokenize(text):
        """ Tokenize text into a list of tokens using built-in methods.
            Parameter
            ---------
            text: string
            Returns
            -------
            list
            """
        text = preprocess(text)
        for p in list(string.punctuation):
            text = text.replace(p, ' ')

        words = text.split()
        return words 



    # Lexical richness measures
    @property
    def ttr(self):
        """ Type-token ratio (TTR) computed as t/w, where t is the number of unique terms/vocab,
            and w is the total number of words.
            (Chotlos 1944, Templin 1957)
        """
        if self.words==0:
          return 1.0
        else:
          return self.terms / self.words


    @property
    def rttr(self):
        """ Root TTR (RTTR) computed as t/sqrt(w), where t is the number of unique terms/vocab,
            and w is the total number of words.
            Also known as Guiraud's R and Guiraud's index.
            (Guiraud 1954, 1960)
        """
        if self.words==0:
          return 1.0
        else:
          return self.terms / sqrt(self.words)


    @property
    def cttr(self):
        """ Corrected TTR (CTTR) computed as t/sqrt(2 * w), where t is the number of unique terms/vocab,
            and w is the total number of words.
            (Carrol 1964)
        """
        if self.words==0:
          return 1.0
        else:
          return self.terms / sqrt(2 * self.words)


    @property
    def Herdan(self):
        """ Computed as log(t)/log(w), where t is the number of unique terms/vocab, and w is the
            total number of words.
            Also known as Herdan's C.
            (Herdan 1960, 1964)
        """
        if self.words==0:
          return 1.0
        else:
          return log(self.terms) / log(self.words) if log(self.words)!=0 else 0


    @property
    def Summer(self):
        """ Computed as log(log(t)) / log(log(w)), where t is the number of unique terms/vocab, and
            w is the total number of words.
            (Summer 1966)
        """
        if self.words==0:
          return 1.0
        else:
          return log(log(self.terms)) / log(log(self.words)) if log(self.words)!=0 else 0


    @property
    def Dugast(self):
        """ Computed as (log(w) ** 2) / (log(w) - log(t)), where t is the number of unique terms/vocab,
            and w is the total number of words.
            (Dugast 1978)
        """
        # raise exception if terms and words count are the same
        if self.words == self.terms:
            #raise ZeroDivisionError('Word count and term counts are the same.')
            return 1.0
        elif self.words==0 and self.terms==0:
            return 0.
        else:
            return (log(self.words) ** 2) / (log(self.words) - log(self.terms)) 


    @property
    def Maas(self):
        """ Maas's TTR, computed as (log(w) - log(t)) / (log(w) * log(w)), where t is the number of
            unique terms/vocab, and w is the total number of words. Unlike the other measures, lower
            maas measure indicates higher lexical richness.
            (Maas 1972)
        """
        if self.words==0:
          return 1.0
        else:
          return (log(self.words) - log(self.terms)) / (log(self.words) ** 2) if (log(self.words) ** 2)!=0 else 0


    def msttr(self, segment_window:int=100, discard:bool=False):
        """ Mean segmental TTR (MSTTR) computed as average of TTR scores for segments in a text.
            Split a text into segments of length segment_window. For each segment, compute the TTR.
            MSTTR score is the sum of these scores divided by the number of segments.
            (Johnson 1944)
            Helper Function
            ---------------
            segment_generator(List, segment_window):
                Split a list into s segments of size r (segment_size).
            Parameters
            ----------
            segment_window: int
                Size of each segment (default=100).
            discard: bool
                If True, discard the remaining segment (e.g. for a text size of 105 and a segment_window
                of 100, the last 5 tokens will be discarded). Default is True.
            Returns
            -------
            float
        """
        if segment_window >= self.words:
            segment_window=(self.words-1)
            #raise ValueError('Window size must be greater than text size of {}. Try a smaller segment_window size.'.format(self.words))
        if segment_window<=0:
            return 0

        if segment_window < 1 or type(segment_window) is float:
            raise ValueError('Window size must be a positive integer.')

        scores = list()
        for segment in segment_generator(self.wordlist, segment_window):
            ttr = len(set(segment)) / len(segment)
            scores.append(ttr)

        if discard: # discard remaining words
            del scores[-1]

        
        mean_ttr = mean(scores)
        #else:
        #    mean_ttr = sum(scores) / len(scores)
        return mean_ttr


    def mattr(self, window_size:int=100):
        """ Moving average TTR (MATTR) computed using the average of TTRs over successive segments
            of a text.
            Estimate TTR for tokens 1 to n, 2 to n+1, 3 to n+2, and so on until the end
            of the text (where n is window size), then take the average.
            (Covington 2007, Covington and McFall 2010)
            Helper Function
            ---------------
            list_sliding_window(sequence, window_size):
                Returns a sliding window generator (of size window_size) over a sequence
            Parameter
            ---------
            window_size: int
                Size of each sliding window.
            Returns
            -------
            float
        """
        #if window_size > self.words:
        #    raise ValueError('Window size must not be greater than text size of {}. Try a smaller window size.'.format(self.words))

        if window_size < 1 or type(window_size) is float:
            raise ValueError('Window size must be a positive integer.')
            
        if window_size>self.words:
            window_size=self.terms//2 if self.terms!=0 else 2 

        if window_size==0:
            return 0.0
        scores = [len(set(window)) / window_size
                  for window in list_sliding_window(self.wordlist, window_size)]

        if len(scores)!=0:
            mattr = mean(scores)
            return mattr
        else:
            return 0.0


    def mtld(self, threshold:float=0.72):
        """ Measure of textual lexical diversity, computed as the mean length of sequential words in
            a text that maintains a minimum threshold TTR score.
            Iterates over words until TTR scores falls below a threshold, then increase factor
            counter by 1 and start over. McCarthy and Jarvis (2010, pg. 385) recommends a factor
            threshold in the range of [0.660, 0.750].
            (McCarthy 2005, McCarthy and Jarvis 2010)
            Parameters
            ----------
            threshold: float
                Factor threshold for MTLD. Algorithm skips to a new segment when TTR goes below the
                threshold (default=0.72).
            Returns
            -------
            float
        """

        def sub_mtld(self, threshold:float, reverse:bool=False):
            """
            Parameters
            ----------
            threshold: float
                Factor threshold for MTLD. Algorithm skips to a new segment when TTR goes below the
                threshold (default=0.72).
            reverse: bool
                If True, compute mtld for the reversed sequence of text (default=False).
            Returns:
                mtld measure (float)
            """
            if reverse:
                word_iterator = iter(reversed(self.wordlist))
            else:
                word_iterator = iter(self.wordlist)

            terms = set()
            word_counter = 0
            factor_count = 0

            for word in word_iterator:
                word_counter += 1
                terms.add(word)
                ttr = len(terms)/word_counter

                if ttr <= threshold:
                    word_counter = 0
                    terms = set()
                    factor_count += 1

            # partial factors for the last segment computed as the ratio of how far away ttr is from
            # unit, to how far away threshold is to unit
            if word_counter > 0:
                factor_count += (1-ttr) / (1 - threshold)

            # ttr never drops below threshold by end of text
            if self.words==0:
                return 0
            if factor_count == 0:
                ttr = self.terms / self.words
                if ttr == 1:
                    factor_count += 1
                else:
                    factor_count += (1-ttr) / (1 - threshold)

            return len(self.wordlist) / factor_count if factor_count!= 0 else 0 

        forward_measure = sub_mtld(self, threshold, reverse=False)
        reverse_measure = sub_mtld(self, threshold, reverse=True)

        #if sys.version_info[0] == 3:
        mtld = mean((forward_measure, reverse_measure))
        #else:
        #    mtld = (forward_measure + reverse_measure) / 2

        return mtld


    def hdd(self, draws:int=42):
        """ Hypergeometric distribution diversity (HD-D) score.
            For each term (t) in the text, compute the probabiltiy (p) of getting at least one appearance
            of t with a random draw of size n < N (text size). The contribution of t to the final HD-D
            score is p * (1/n). The final HD-D score thus sums over p * (1/n) with p computed for
            each term t. Described in McCarthy and Javis 2007, p.g. 465-466.
            (McCarthy and Jarvis 2007)
            Parameters
            ----------
            draws: int
                Number of random draws in the hypergeometric distribution (default=42).
            Returns
            -------
            float
        """
        if self.terms < 42:
            suggestion = self.words // 2
        else:
            suggestion = 42
        if self.words < draws:
            draws=self.words
            #raise ValueError('Number of draws should be less than the total sample size of {0}. Try a draw value smaller than {0}, e.g. hdd(draws={1}.)'.format(self.words, suggestion))
        if draws<=0:
            return 0
        if draws < 1 or type(draws) is float:
            raise ValueError('Number of draws must be a positive integer. E.g. hdd(draws={})'.format(suggestion))

        term_freq = Counter(self.wordlist)

        term_contributions = [(1 - hypergeom.pmf(0, self.words, freq, draws)) / draws
                              for term, freq in term_freq.items()]

        return sum(term_contributions)
    
    def get_all_metrics(self):
      """
      TODO:Documentation
      """
      return (round(self.ttr,3),round(self.rttr,3),round(self.cttr,3),
                round(self.Herdan,3),round(self.Summer,3),round(self.Dugast,3),
               round(self.mattr(window_size=5),3),round(self.msttr(segment_window=5),3),
               round(self.mtld(),3),round(self.hdd(),3))
        


    def __str__(self):
        return ' '.join(self.wordlist)


    def __repr__(self):
        return 'LexicalRichness(words={0}, terms={1}, tokenizer={2}, wordlist={3}, string="{4}")'.format(
            self.words, self.terms, self.tokenizer, self.wordlist, ' '.join(self.wordlist)
        )