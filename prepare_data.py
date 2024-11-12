from datasets import load_dataset
import json
import os


contentX = []
contentY = []
with open("data/testX.txt", "r") as fileX:
    for line in fileX:
        contentX.append(line.strip())
    
    
with open("data/testY.txt", "r") as fileY:
    for line in fileY:
        contentY.append(line.strip())
        
data_len = len(contentX)

data_file = "data/data_mul_doc_summary.json"
with open(data_file, "w") as file:
    for i in range(data_len):
        X = contentX[i]
        Y = contentY[i]
        json.dump({'idx': i, 'x': X, 'y': Y}, file)
        file.write('\n')
        if i > 20:
            break
    

