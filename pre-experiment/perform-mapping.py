import os
import sys
from tqdm import tqdm

mappingData = None
dir = sys.argv[1]

with open(f"{dir}/mapping.txt", "r") as mappingFile:
    lines = mappingFile.readlines()
    for line in tqdm(lines):
        mappingData = line.strip().split(" : ")
        frm = mappingData[0].replace("_", "\\_")
        to = mappingData[1].replace("_", "\\_")
        os.system(f"sed -i 's/{frm}/{to}/g' {dir}/similarity-scores.txt")
