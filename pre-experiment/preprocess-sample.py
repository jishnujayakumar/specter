import pickle as pkl
import sys
import json
from helpers import *
from tqdm import tqdm
import os

dir = sys.argv[1]
testDocs = data = None

ppdDir = f"{dir}/preProcessedData"

with open(f"{ppdDir}/data.json", "r") as dataF:
    data = json.load(dataF)

# Create a smaller dataset [nsample docs] data-small.json for testing purposes
nsamples = int(sys.argv[2])
sampletrainDocIds = list(data.keys())[:nsamples]
dataSamples = {}
count = -1
for fromDocID in tqdm(sampletrainDocIds):
    if fromDocID in sampletrainDocIds:
        toDocsCitation = data[fromDocID]
        result = {}
        for toDocID in toDocsCitation.keys():
            if toDocID in sampletrainDocIds:
                result[toDocID] = toDocsCitation[toDocID]
        if result:
            dataSamples[fromDocID] = result
            count += 1

    if count == nsamples:
        break

sampleMetadata = {}
with open(f"{ppdDir}/metadata.json", "r") as dataF:
    data = json.load(dataF)
    count = -1
    for k, v in data.items():
        if k in sampletrainDocIds:
            sampleMetadata[k] = data[k]
            count += 1

        if count == nsamples:
            break

ppdDir = f"{ppdDir}/sampled"
os.system(f"mkdir -p {ppdDir}")

with open(f"{ppdDir}/data-{nsamples}-samples.json", "w") as outF:
    json.dump(dataSamples, outF, indent=2)

with open(f"{ppdDir}/metadata-{nsamples}-samples.json", "w") as outF:
    json.dump(sampleMetadata, outF, indent=2)

save2Pickle(sampletrainDocIds, f"{ppdDir}/sampletrainDocID-Set.pkl")