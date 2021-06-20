"""
Created on Tue Jan 28 21:30:00 2020

@author: paheli
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import spacy

#import os

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
    """
     Initilize the text summarizer.
     Words that have a frequency term lower than min_cut 
     or higer than max_cut will be ignored.
    """
    self._min_cut = min_cut
    self._max_cut = max_cut 
    self._stopwords = set(stopwords.words('english') + list(punctuation))
    self.NLP = spacy.load('en_core_web_sm')

  def _compute_frequencies(self, word_sent):
    """ 
      Compute the frequency of each of word.
      Input: 
       word_sent, a list of sentences already tokenized.
      Output: 
       freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in list(freq):
      freq[w] = freq[w]/m
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    """
      Return a list of n sentences 
      which represent the summary of text.
    """
    doc = self.NLP(text)
    sents = [s.text for s in doc.sents]
    if n > len(sents): n = len(sents)
    
    word_sent = [word_tokenize(s.lower()) for s in sents]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self._rank(ranking, n)    
    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)


'''
def word_counted_summary(summary,wordcount):   
    count_sum = 0
    final_summary = []
    for s in summary:
        parser2 = PlaintextParser.from_string(s, Tokenizer(LANGUAGE))
        count_sum+=len(parser2.tokenize_words(s))
        
        if count_sum<wordcount:
            final_summary.append(s)
        else:
            break
    
    return final_summary
'''
     

'''
fs = FrequencySummarizer()
path = ''
pathw = ''

for f in os.listdir(path):
    fw = os.path.join(pathw,f)
    final = open(fw,"w")
    fr = open(os.path.join(path,f),"r")
    #document = ''.join(open(fr,"r").readlines()
    #document = ' '.join(document.strip().split('\n'))
    doc = []
    for line in fr.readlines():
        line = line.rstrip("\n")
        doc.append(line)
    document = ' '.join(doc)
    
    summary = []
    for s in fs.summarize(document, SENTENCES_COUNT):
       summary.append(s)
    final_summary = sentCutOff(summary,SUMMARYLEN[f])
    
    final.write("\n".join(final_summary))
    final.close()
'''