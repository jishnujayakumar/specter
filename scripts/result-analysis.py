import logging
import os
import sys
import jsonlines
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import mean_squared_error, f1_score


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def computeEmbeddingSimilarity(filePath):
    result = {}
    thresholdP = 0.5

    with jsonlines.open(f"{filePath}/embeddings-output-gold-docs.jsonl", 'r') as embeddingResults:
        for res in embeddingResults.iter():
            result[res['paper_id']] = res['embedding']

    frmArr = []
    toArr = []
    goldSimArr = []
    cosineArr = []
    corr = None
    pearsonOutputF = f"{filePath}/pearson-correlation-gold-sim-vs-cosine-sim.png"

    with open(f"{filePath}/similarity-scores.txt", "r") as simScoreF:
        lines = simScoreF.readlines()
        for line in tqdm(lines):
            line = list(filter(None, line.strip().split("\t")))
            frmArr.append(line[0])
            toArr.append(line[1])
            goldSimArr.append(float(line[2]))

            cosineArr.append(
                round(cosine_similarity(
                    result[line[0]],
                    result[line[1]]
                ), 2)
            )
        df = pd.DataFrame()
        df['fromDocID'] = frmArr
        df['toDocID'] = toArr
        df['goldSimilarityValue'] = goldSimArr
        df['cosineSimilarityValue'] = cosineArr

        corr = df.corr(method='pearson')
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt='.4f', 
                    cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
        ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")
        plt.savefig(pearsonOutputF, bbox_inches='tight', pad_inches=0.0)
        df.to_csv(f"{filePath}/sim-vs-cosine-sim-result.csv", index=False)

    true = [1 if score > thresholdP else 0 for score in goldSimArr]
    pred = [1 if score > thresholdP else 0 for score in cosineArr]

    resultMetrics = {
        "f1-score": round(f1_score(true, pred), 2),
        "mse": round(mean_squared_error(true, pred), 2),
        "pearson-corr": pearsonOutputF,
        "thresholdP": thresholdP,
        "goldScore": goldSimArr,
        "cosineSimScore": cosineArr,
        "trueLabel": true,
        "predLabel": pred
    }

    os.system(f'echo "{json.dumps(resultMetrics)}" >> {filePath}/result-metrics.json')


dir = sys.argv[1]
ELECTER_DIR = os.environ['ELECTER_DIR']
filePath = f"{ELECTER_DIR}/{dir}"

computeEmbeddingSimilarity(filePath)
