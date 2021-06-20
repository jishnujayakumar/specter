"""
Created on Sun Mar  1 01:29:24 2020

@author: paheli
"""

import os
import json
import math
import spacy
import string
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from sumy.summarizers.lsa import LsaSummarizer 
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

from OtherAlgos.freqsum import FrequencySummarizer
from OtherAlgos.DSDR.dsdr import DSDR
from sklearn.feature_extraction.text import TfidfVectorizer

from split_sentences import custom_splitter

import logging

#PATH = '../samples/test/judgement'
#OUTDIR = '../samples/summaries'
#SUMMARYSIZEJSON = '../samples/test/test_word_count.json'
#AVGWORDPERSENT = 27.5

#DIR = 'path to doc folder'

ELECTER_DIR = os.environ['ELECTER_DIR']
logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.INFO, filename=f"{ELECTER_DIR}/legal-data-dsdr-summarized/Summ-CustomTokenizer.log"
        )


PATH = f"{ELECTER_DIR}/legal-data-dsdr-summarized/original-castext-without-summarization"
OUTDIR = f"{ELECTER_DIR}/legal-data-dsdr-summarized/casetext"
# SUMMARYSIZEJSON = open('file containing summary length in no.of words.txt',"r")
#file containing summary length in no.of words.txt format : filename<tab>summary-len-count
SUMMARYSIZEJSON = open(f"{ELECTER_DIR}/legal-data-dsdr-summarized/file-to-summary-size.txt","r")

AVGWORDPERSENT = 17


SUMY = False # Default was True, for DSDR SUMY=False as suggested by Paheli
UNSUPERVISED = True #for DSDR

LANG = 'english'


try: os.mkdir(OUTDIR)
except: pass


#NLP = custom_splitter()


def countWord(text):
    doc = NLP(text)
    tokens = [t.text for t in doc]
    tokens = [t for t in tokens if len(t.translate(t.maketrans('', '', string.punctuation + string.whitespace))) > 0] # + string.digits    
    return len(tokens)


def getNoSents(fn):
        return math.ceil(SUMMARYLEN[fn] / AVGWORDPERSENT)



def sentCutoff(summary, size):
        newsumm = []
        currsize = 0
        for sent in summary:
                cnt = countWord(str(sent))
                if currsize + cnt > size:
                        break
                
                currsize += cnt
                newsumm.append(sent)
        else:
                logging.info(
                        f"LESS SENTS IN SUMMARY: 'Words Required: {size} | Words in Summary: {currsize}"
                )

        return newsumm


def getSumySummaries(fn, Summarizer, outpath):
#        doc = PlaintextParser.from_file(os.path.join(PATH, fn), Tokenizer(LANG)).document
        doc = PlaintextParser.from_file(os.path.join(PATH, fn), customTokenizer()).document
        
        summr = Summarizer(stemmer)
        summr.stop_words = get_stop_words(LANG)
        
        multiplier = 3 if Summarizer in [SumBasicSummarizer] else 1
        summary = summr(doc, multiplier * getNoSents(fn))
        summary = sentCutoff(summary, SUMMARYLEN[fn])
        
        with open(os.path.join(outpath, fn), 'w') as fout:
                for sent in summary:
                        print(str(sent), file = fout)
                        
        return summary




class customTokenizer:
        def to_words(self, text):
                return [s.text for s in NLP(text) if len(s.text.translate(str.maketrans('', '', string.punctuation))) > 0]
        def to_sentences(self, text):
                return [s.text for s in NLP(text).sents]


#%% MAIN
SUMMARYLEN = {}
for line in SUMMARYSIZEJSON.readlines():
    line = line.rstrip("\n")
    ls = line.split("\t")
    SUMMARYLEN[ls[0]] = int(ls[1].rstrip("\n"))
#NLP = spacy.load('en_core_web_md')
NLP = custom_splitter()
fileslist = tqdm(next(os.walk(PATH))[2])

#sumy modules
if SUMY:
        stemmer = Stemmer(LANG)
        SummarizerList = [LsaSummarizer, LexRankSummarizer, SumBasicSummarizer, ReductionSummarizer, LuhnSummarizer]
#        SummarizerList = [SumBasicSummarizer]
        
        
        for summarizer in SummarizerList:
                outpath = os.path.join(OUTDIR, summarizer.__name__)
                try: os.mkdir(outpath)
                except: os.system('rm %s/*'%outpath)
                
                print('\n', summarizer.__name__, flush = True)
                for fn in fileslist:
                        summaries = getSumySummaries(fn, summarizer, outpath)
                
                
if UNSUPERVISED:
        
        # freqsum
        # outpath = os.path.join(OUTDIR, "FreqSum") # commented for ease of use with specter 
        outpath = OUTDIR
        try: os.mkdir(outpath)
        except: os.system('rm %s/*'%outpath)
        
        print('\n', "FreqSum", flush = True)
        fsummr = FrequencySummarizer()
        for fn in fileslist:
                with open(os.path.join(PATH, fn)) as fp:
                        document = fp.read().replace('\n', ' ')
                summary = fsummr.summarize(document, getNoSents(fn))
                summary = sentCutoff(summary, SUMMARYLEN[fn])
                with open(os.path.join(outpath, fn), 'w') as fout:
                        for sent in summary:
                                print(str(sent), file = fout)
        
        
        
        # DSDR
        outpath = os.path.join(OUTDIR, "DSDR")
        try: os.mkdir(outpath)
        except: os.system('rm %s/*'%outpath)
        
        print('\n', "DSDR", flush = True)
        for fn in fileslist:
                with open(os.path.join(PATH, fn)) as fp:
                        document = NLP(fp.read().replace('\n', ' '))
                        
                sentences = [s.text for s in document.sents if len(s.text.strip()) > 10]
                tfidf = TfidfVectorizer()
                normalized_matrix = tfidf.fit_transform(sentences)
                
                summary_idx = DSDR.lin(normalized_matrix.toarray(), 2 * getNoSents(fn), 0.1)
                summary = [sentences[i] for i in summary_idx]
                summary = sentCutoff(summary, SUMMARYLEN[fn])
                with open(os.path.join(outpath, fn), 'w') as fout:
                        for sent in summary:
                                print(str(sent), file = fout)
        
