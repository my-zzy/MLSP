import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

with open("data/toenglish.json", 'r', encoding='utf-8') as f:
    toenglish = json.load(f)


for k in range(1,6,1):
    folder_path = "data/filtered_{:d}".format(k)

    file_names = os.listdir(folder_path)

    # print(file_names)

    for i in file_names:
        if i[:2] != "混合":
            continue
        
        data = pd.read_csv(folder_path + "/" + i, header=None)
        x = data[0]
        y = data[1]
        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.title(toenglish[i[:2]])
        plt.xticks(np.arange(0, 12, 1))
        plt.show()
