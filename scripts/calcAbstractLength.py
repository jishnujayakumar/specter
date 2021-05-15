import sys
import ijson
import pickle
import numpy as np

absLen = np.array([]).astype(int)
suffix = 'abstractLength.pkl'

# This is not the standard way to use ijson (ijson.items is yet to be implemented), this is a temp script for calculating abstract lengths
# Assumming entered filename is valid
inpFile = sys.argv[1]
outFilePrefix = inpFile.split("/")[-1].split(".")[0]
counter = 1

for prefix, type_of_object, value in ijson.parse(open(inpFile)):
    if "abstract" in prefix:
        print(f"\r{counter}")
        counter += 1
        absLen = np.append(absLen, len(str(value).split(" ")))

stats = {
            "minLen": np.min(absLen),
            "maxLen": np.max(absLen),
            "meanLen": np.mean(absLen),
            "stdLen": np.std(absLen)
        }

with open(f"{outFilePrefix}_{suffix}",'wb') as outfile:   
    pickle.dump({
        "absLenList": absLen,
        "stats" : stats
        }, outfile)

print(stats)
