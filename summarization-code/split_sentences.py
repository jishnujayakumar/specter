import spacy
from spacy.attrs import ORTH
import os
from tqdm import tqdm
import re


def custom_sentencizer(doc):
    ''' Look for sentence start tokens by scanning for periods only. '''
    for i, token in enumerate(doc[:-2]):  # The last token cannot start a sentence
        if token.text == ".":
            #doc[i+1].is_sent_start = True
            pass
        else:
            doc[i+1].is_sent_start = False  # Tell the default sentencizer to ignore this token
    return doc


def custom_splitter(text = None):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(custom_sentencizer, before = "parser")

    special_cases = {"Rs.": "Rs.", "No.": "No.", "no.": "No.", "vs.": "vs", "i.e.": "i.e.", "viz.": "viz.", "M/s.": "M/s.", "Mohd.": "Mohd.", "Ex.": "exhibit", "Art." : "article", "Arts." : "articles", "S.": "section", "s.": "section", "ss.": "sections", "u/s.": "section", "u/ss.": "sections", "art.": "article", "arts.": "articles", "u/arts." : "articles", "u/art." : "article"}
    
    for case, orth in special_cases.items():
    	nlp.tokenizer.add_special_case(case, [{ORTH: orth}])
    
    # ADDED THIS LINE:
    if text is None: return nlp
	
    #text = text.strip()
    #print (text)
    text = text.replace('\n', ' ')
    #text = re.sub(' +', ' ', text)
    
    
    
    parsed = nlp(text)
    
    sentences = []
    
    for sent in parsed.sents:
        sentences.append(sent.text)
    
    return sentences
