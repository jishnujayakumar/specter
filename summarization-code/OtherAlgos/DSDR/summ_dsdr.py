from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from dsdr import DSDR
import os
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
#num_of_sent = int(sys.argv[1])
#document = open(sys.argv[2],"r").read()
#final = open(sys.argv[3],"w")
path = "F:\\Summarization\\INDIA-TEST-DATA\\processed\\"
pathw = "'F:\Summarization\\WRAP_UP_ICAIL\\India\\summaries-preproc\\DSDR"
LANGUAGE = "english"

def word_count(file):
    parser = PlaintextParser.from_file(file, Tokenizer(LANGUAGE))
    #SENTENCES_COUNT = 0
    WORD_COUNT = 0
    fr = open(file,"r")
    for line in fr.readlines():
        line = line.rstrip("\n")
        WORD_COUNT+=len(parser.tokenize_words(line))
    return WORD_COUNT

'''
def word_counted_summary(summary_indx,sentences,wordcount):
    summary = []
    for i in summary_indx:
        summary.append(sentences[i])
        
    count_sum = 0
    final_summary = []
    for s in summary:
        parser2 = PlaintextParser.from_string(s, Tokenizer(LANGUAGE))
        count_sum+=len(parser2.tokenize_words(s))
        
        if count_sum<=wordcount:
            final_summary.append(s)
        else:
            break
    
    return final_summary
'''


def generate_summary(sentences, num_of_sent)
    '''
    fw = os.path.join(pathw,fn)
    final = open(fw,"w")
    fr = open(os.path.join(path,f),"r")
    #document = ''.join(open(fr,"r").readlines()
    #document = ' '.join(document.strip().split('\n'))
    doc = []
    for line in fr.readlines():
        line = line.rstrip("\n")
        doc.append(line)
    document = ' '.join(doc)
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
    '''
    ################################################################################
    #WORD_COUNT = word_count(os.path.join(path,f))
    #num_of_sent = len(sentences)
    #num_of_sent = int(len(sentences)/3)
    ################################################################################
    
    c = CountVectorizer()
    bow_matrix = c.fit_transform(sentences)
    normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)
    
    summary = DSDR.lin(normalized_matrix.toarray(),num_of_sent,0.1)
    ################################################################################
#    for i in summary:
#    	final.writelines(sentences[i]+'\n')
#    final.close()
    #final_summary = word_counted_summary(summary,sentences,WORD_COUNT/3)
    #final.write("\n".join(final_summary))
    #final.close()
    
    ################################################################################


