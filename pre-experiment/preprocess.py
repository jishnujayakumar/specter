import os
import sys
from tqdm import tqdm
import pickle as pkl
import json
from helpers import *
from collections import defaultdict
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

mappingData = None
dir = sys.argv[1]

# 1. Map similarity-scores.txt docIDs using mapper.txt 
logging.info("Mapping similarity-scores.txt docIDs using mapper.txt")
with open(f"{dir}/mapping.txt", "r") as mappingFile:
    lines = mappingFile.readlines()
    for line in tqdm(lines):
        mappingData = line.strip().split(" : ")
        frm = mappingData[0].replace("_", "\\_")
        to = mappingData[1].replace("_", "\\_")
        os.system(f"sed -i 's/{frm}/{to}/g' {dir}/similarity-scores.txt")


# 2. Extract test docIDs from similarity-scores.txt
logging.info("Extracting test docIDs from similarity-scores.txt")
testDocIDs = set([])
pklDir = f"{dir}/preProcessedData"
os.system(f"mkdir -p {pklDir}")
with open(f"{dir}/similarity-scores.txt", "r") as mappingFile:
    lines = mappingFile.readlines()
    for line in tqdm(lines):
        strippedData = line.strip().split("\t")
        scoreData = list(filter(None, strippedData))
        testDocIDs.update(scoreData[:-1])
    save2Pickle(testDocIDs, f"{pklDir}/testDocsSet.pkl")

    os.system(f"rm {pklDir}/test.txt")
    saveDocIDsToTXT("\n".join(testDocIDs), f"{pklDir}/test.txt")


# Create data.json [training set] excluding test DocIDs from 3
logging.info("Creating data.json [training set] excluding test DocIDs from 3")
data = {}
"""
Since precedent-citation.txt only contains infdormation about positive citations
Here only positive samples are considered
TODO: Discuss with team regarding the formulation of hard-negative signals
"""
posSample = {"count": 5}

with open(f"{dir}/precedent-citation.txt", "r") as citationsInfoF:
    citations = citationsInfoF.readlines()
    for citation in tqdm(citations):
        docsIDs = citation.strip().split(" : ")
        frm, to = docsIDs[0], docsIDs[1]
        # Exclude test samples from data.json [training set]
        if frm not in testDocIDs and to not in testDocIDs:
            if frm in data:
                docCiteData = data[frm]
                docCiteData[to] = posSample
                data[frm] = docCiteData
            else:
                data[frm] = {to: posSample}
    with open(f"{pklDir}/postive_data.json", "w") as outF:
        json.dump(data, outF, indent=2)

negSample = {"count": 1}
data2 = {}
dkeys = data.keys()
for P1 in tqdm(dkeys):
    if P1 in dkeys:
        P2s = data[P1].keys()
        for P2 in P2s:
            if P2 in dkeys:
                P3s = data[P2].keys()
                for P3 in P3s:
                    if P3 not in P2s:  # condition for hard sample
                        data2[P1] = data[P1]
                        data2[P3] = {P1: negSample}

with open(f"{pklDir}/data.json", "w") as outF:
    json.dump(data2, outF, indent=2)

# Preprocess casetext
logging.info("Preprocessing casetext")
casetextDir = f"{dir}/casetext"
metadata = {}
vocabTokens = defaultdict(int)
headerTokens = defaultdict(int)

filesEncountered = 0
docs = os.listdir(casetextDir)
nDocs = len(docs)

trainDocs = []
valDocs = []

for doc in tqdm(docs):
    filesEncountered += 1
    docID = doc.split(".")[0]
    filePath = f"{casetextDir}/{doc}"
    casetext = getCaseTextContent(filePath).lower()

    # Following uses the "1. " as the seperator for header and body 
    # Doesn't work properly, hence going with method-2
    # # Find the first occurence
    # splitIndex = casetext.find("1. ")  
    # header = casetext[:splitIndex].strip()
    # body = casetext[splitIndex:].strip()

    # Method-2 Seperation using "The Judgment was delivered"
    s=f"grep -n 'The Judgment was delivered' {filePath}"
    # print(doc,os.popen(s).read().split(":"))
    # splitLineNum = int(os.popen(s).read().split(":")[0])
    # print(splitLineNum)
    # header, body = splitbyLineNumber(filePath, splitLineNum)
    body = casetext
    metadata[docID] = {
        "paper_id": docID,
        "title": body,
        "abstract": ""
    }

    if filesEncountered/nDocs < 0.8:
        trainDocs.append(docID)
    else:
        valDocs.append(docID)

    for token in body.replace("\n", " ").split(" "):
        vocabTokens[token] += 1
    # for headerToken in header.replace("\n", " ").split(" "):
    #     headerTokens[headerToken] += 1

saveDocIDsToTXT("\n".join(trainDocs), f"{pklDir}/train.txt")
saveDocIDsToTXT("\n".join(valDocs), f"{pklDir}/val.txt")

vocabTokens = sortByValues(vocabTokens)
headerTokens = sortByValues(headerTokens)

logging.info("Saving preprocessed artifacts")

save2Pickle(vocabTokens, f"{pklDir}/vocabTokens.pkl")
save2Pickle(headerTokens, f"{pklDir}/headerTokens.pkl")

vocabTokens = list(filter(None, vocabTokens.keys()))
headerTokens = list(filter(None, headerTokens.keys()))

ELECTER_DIR = os.environ['ELECTER_DIR']

vocabDir = f"{ELECTER_DIR}/pre-experiment/legal-data-vocab/"

os.system(f"mkdir -p {vocabDir} && \
    cp {ELECTER_DIR}/data/vocab/non_padded_namespaces.txt {vocabDir}")

saveTokens(vocabTokens, f"{vocabDir}/tokens.txt")
saveTokens(headerTokens, f"{vocabDir}/header.txt")


with open(f"{pklDir}/metadata.json", "w") as outF:
    json.dump(metadata, outF, indent=2)
