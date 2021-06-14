import logging
import os
import sys
import jsonlines
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def computeEmbeddingSimilarity(filePath):
    result = {}

    with jsonlines.open(f"{filePath}/embeddings-output-gold-docs.jsonl", 'r') as embeddingResults:
        for res in embeddingResults.iter():
            result[res['paper_id']] = res['embedding']

    frmArr = []
    toArr = []
    goldSimArr = []
    cosineArr = []

    with open(f"{filePath}/similarity-scores.txt", "r") as simScoreF:
        lines = simScoreF.readlines()
        for line in tqdm(lines):
            line = list(filter(None, line.strip().split("\t")))
            frmArr.append(line[0])
            toArr.append(line[1])
            goldSimArr.append(line[2])

            cosineArr.append(
                cosine_similarity(
                    result[line[0]],
                    result[line[1]]
                )
            )
        df = pd.DataFrame()
        df['fromDocID'] = frmArr
        df['toDocID'] = toArr
        df['goldSimilarityValue'] = goldSimArr
        df['cosineSimilarityValue'] = cosineArr

        fig, ax = plt.subplots()
        sns.heatmap(df['goldSimilarityValue'].corr(df['cosineSimilarityValue'],
                    method='pearson'), annot=True, fmt='.4f',
                    cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        plt.savefig(f"{filePath}/pearson-correlation-gold-sim-vs-cosine-sim.png", bbox_inches='tight', pad_inches=0.0)
        df.to_csv(f"{filePath}/sim-vs-cosine-sim-result.csv", index=False)


dir = sys.argv[1]
ELECTER_DIR = os.environ['ELECTER_DIR']
filePath = f"{ELECTER_DIR}/{dir}"

computeEmbeddingSimilarity(filePath)
