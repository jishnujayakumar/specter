import os
import sys
import json
import random 
from tqdm import tqdm
from faker import Faker

fake = Faker()
random.seed(1)

ndocs = int(sys.argv[1])

os.system("mkdir -p generated_data")

# Create metadata.json
metadata = {}

for docID in tqdm(range(ndocs)):
    metadata[docID] = {
      "paper_id": str(docID),
    #   "abstract": fake.text(),
      "title": fake.sentence()
    }

    ctgy = None

    if docID < ndocs*0.6:
        ctgy = "train"
    elif docID > ndocs*0.6 and docID < ndocs*0.75:
        ctgy = "val"
    else:
        ctgy = "test"

    os.system(f"echo {docID} >> generated_data/{ctgy}.txt")

with open("generated_data/metadata.json", "w") as out_file:
    json.dump(metadata, out_file, indent=4)

# Create data.json
data = {}
for rowID in tqdm(range(ndocs)):
    docCount = {}
    for colID in range(ndocs):
        p = random.uniform(0, 1)
        if p < 0.25:
            docCount[colID] = {"count": 1}
        elif p >= 0.25 and p < 0.6:
            docCount[colID] = {"count": 5}
    data[rowID] = docCount

with open("generated_data/data.json", "w") as out_file:
    json.dump(data, out_file, indent=4)
