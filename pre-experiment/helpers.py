import os
import pickle as pkl
from collections import OrderedDict
from tqdm import tqdm

def save2Pickle(data, filepath):
    os.system(f"touch {filepath}")
    with open(filepath, "wb") as outF:
        pkl.dump(data, outF)


def getCaseTextContent(caseTextDocFilePath):
    with open(caseTextDocFilePath) as inpF:
        return inpF.read()


def sortByValues(dictionary):
    sortedList = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    dictionary = OrderedDict()
    for i in sortedList:
        dictionary[i[0]] = i[1]
    return dictionary


def saveTokens(tokens, file):
    os.system(f"rm -rf {file}")
    if "token" in file:
        for i in ["@@UNKNOWN@@", "[cls]", "[CLS]"]:
            os.system(f"echo \"{i}\" >> {file}")
    for token in tqdm(tokens):
        with open(file, "a") as F:
            F.write(f"{token}\n")


def saveDocIDsToTXT(docIDString, filepath):
    with open(filepath, "w") as txtF:
        txtF.write(docIDString)
