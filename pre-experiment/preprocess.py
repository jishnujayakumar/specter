import argparse
import os
from tqdm import tqdm
import json
from helpers import *
from collections import defaultdict
import logging
import nltk

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Preprocess legal-dataset \
    [Courtesy: IIT-KGP]')

parser.add_argument('--dir', type=str, default="legal-data",
                    help='legal dataset directory relative path \
                        w.r.t. ELECTER_DIR')
parser.add_argument('--trainDataPercent', type=float, default=0.8,
                    help='Percentage (scale is 1 in lieu of 100) \
                        of total data to be used as train set')
parser.add_argument('--samplePercent', type=float, default=1,
                    help='Percentage (scale is 1 in lieu pf 100) \
                        of total data to be as dataset')
args = parser.parse_args()

mappingData = None
dir = args.dir
trainP = args.trainDataPercent
valP = 1-trainP
testP = 0
ELECTER_DIR = os.environ['ELECTER_DIR']

# 1. Map similarity-scores.txt docIDs using mapper.txt
logging.info("Mapping similarity-scores.txt docIDs using mapper.txt")
with open(f"{dir}/mapping.txt", "r") as mappingFile:
    lines = mappingFile.readlines()
    for line in tqdm(lines):
        mappingData = line.strip().split(" : ")
        frm = mappingData[0].replace("_", "\\_")
        to = mappingData[1].replace("_", "\\_")
        os.system(f"sed -i 's/{frm}/{to}/g' \
        {dir}/similarity-scores.txt")


# 2. Extract test docIDs from similarity-scores.txt
logging.info("Extracting test docIDs from similarity-scores.txt")
goldScoreDocIDs = set([])
pklDir = f"{dir}/preProcessedData"
goldScoreDir = f"{dir}/Gold-Score-Docs"
caseTextDir = f"{dir}/casetext"

os.system(f"mkdir -p {pklDir} {goldScoreDir}")
with open(f"{dir}/similarity-scores.txt", "r") as mappingFile:
    lines = mappingFile.readlines()
    for line in tqdm(lines):
        strippedData = line.strip().split("\t")
        scoreData = list(filter(None, strippedData))
        goldScoreDocIDs.update(scoreData[:-1])
    save2Pickle(goldScoreDocIDs, f"{pklDir}/goldScoreDocIDs-Set.pkl")

goldDocFiles = " ".join([
    f"{caseTextDir}/{goldScoreDocID}.txt"
    for goldScoreDocID in goldScoreDocIDs])
os.system(f"mv {goldDocFiles} {goldScoreDir}/")

metadata = {}
nltk.download("punkt")
for goldScoreDocID in goldScoreDocIDs:
    filePath = f"{goldScoreDir}/{goldScoreDocID}.txt"
    casetext = getCaseTextContent(filePath).lower()

    words = nltk.word_tokenize(casetext)
    new_words = [word for word in words if word.isalnum()]
    body = " ".join(new_words)

    metadata[goldScoreDocID] = {
        "paper_id": goldScoreDocID,
        "title": body,
        "abstract": ""
    }
with open(f"{goldScoreDir}/metadata.json", "w") as outF:
    json.dump(metadata, outF, indent=2)

saveDocIDsToTXT("\n".join(goldScoreDocIDs), f"{goldScoreDir}/gold-docs.txt")

# Create citation-adj-list.json [training set] excluding test DocIDs from 3
logging.info("Creating citation-adj-list.json [training set] excluding \
test DocIDs from 3")
citationAdjList = defaultdict(list)
"""
Since precedent-citation.txt only contains infdormation about positive
citations. Here only positive samples are considered
TODO: Discuss with team regarding the formulation of hard-negative signals
"""

casetextDir = f"{dir}/casetext"
docs = os.listdir(casetextDir)
nDocs = int(len(docs)*args.samplePercent)
docs = [doc.split(".")[0] for doc in docs[:nDocs]]

with open(f"{dir}/precedent-citation.txt", "r") \
        as citationsInfoF:
    citations = citationsInfoF.readlines()
    for citation in tqdm(citations):
        docsIDs = citation.strip().split(" : ")
        frm, to = docsIDs[0], docsIDs[1]
        # Exclude test samples from data.json [training set]
        if frm not in goldScoreDocIDs and to not in goldScoreDocIDs and \
                frm in docs and to in docs:
            citationAdjList[frm].append(to)
    with open(f"{pklDir}/citation-adj-list.json", "w") as outF:
        json.dump(citationAdjList, outF, indent=2)

posSample = {"count": 5}
hardNegSample = {"count": 1}

data = {}
P1s = citationAdjList.keys()
for P1 in tqdm(P1s):
    P2s = citationAdjList[P1]
    result = {}
    for P2 in P2s:
        result[P2] = posSample
        if P2 in P1s:
            P3s = citationAdjList[P2]
            for P3 in P3s:
                if P3 not in P2s:  # condition for hard negative sample
                    result[P3] = hardNegSample
    data[P1] = result

with open(f"{pklDir}/data.json", "w") as outF:
    json.dump(data, outF, indent=2)

# Preprocess casetext
logging.info("Preprocessing casetext")
metadata = {}
vocabTokens = defaultdict(int)
headerTokens = defaultdict(int)

filesEncountered = 0

trainDocs = []
valDocs = []
testDocs = []

for docID in tqdm(docs):
    filesEncountered += 1
    filePath = f"{casetextDir}/{docID}.txt"
    casetext = getCaseTextContent(filePath).lower()

    # Following uses the "1. " as the seperator for header and body 
    # Doesn't work properly, hence going with method-2
    # # Find the first occurence
    # splitIndex = casetext.find("1. ")  
    # header = casetext[:splitIndex].strip()
    # body = casetext[splitIndex:].strip()

    # Method-2 Seperation using "The Judgment was delivered"
    # s=f"grep -n 'The Judgment was delivered' {filePath}"
    # print(doc,os.popen(s).read().split(":"))
    # splitLineNum = int(os.popen(s).read().split(":")[0])
    # print(splitLineNum)
    # header, body = splitbyLineNumber(filePath, splitLineNum)

    """
    TODO: Get a way to differentiate header and body for casetext docs
    In lieu of it right now all content is taken as input, i.e. body
    """

    words = nltk.word_tokenize(casetext)
    new_words = [word for word in words if word.isalnum()]
    body = " ".join(new_words)

    metadata[docID] = {
        "paper_id": docID,
        "title": body,
        "abstract": ""
    }

    percentage = filesEncountered/nDocs

    if percentage <= trainP:
        trainDocs.append(docID)
    elif percentage > trainP and percentage <= trainP + valP:
        valDocs.append(docID)
    else:
        testDocs.append(docID)

    for token in body.replace("\n", " ").split(" "):
        vocabTokens[token] += 1
    # for headerToken in header.replace("\n", " ").split(" "):
    #     headerTokens[headerToken] += 1

saveDocIDsToTXT("\n".join(trainDocs), f"{pklDir}/train.txt")
saveDocIDsToTXT("\n".join(valDocs), f"{pklDir}/val.txt")
saveDocIDsToTXT("\n".join(testDocs), f"{pklDir}/test.txt")

vocabTokens = sortByValues(vocabTokens)
headerTokens = sortByValues(headerTokens)

logging.info("Saving preprocessed artifacts")

save2Pickle(vocabTokens, f"{pklDir}/vocabTokens.pkl")
save2Pickle(headerTokens, f"{pklDir}/headerTokens.pkl")

vocabTokens = list(filter(None, vocabTokens.keys()))
headerTokens = list(filter(None, headerTokens.keys()))

vocabDir = f"{dir}/legal-data-vocab/"

os.system(f"mkdir -p {vocabDir} && \
    cp {ELECTER_DIR}/data/vocab/non_padded_namespaces.txt {vocabDir}")

saveTokens(vocabTokens, f"{vocabDir}/tokens.txt")
saveTokens(headerTokens, f"{vocabDir}/header.txt")


with open(f"{pklDir}/metadata.json", "w") as outF:
    json.dump(metadata, outF, indent=2)
