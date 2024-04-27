import math
import numpy as np
import pandas as pd
import pywt
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, confusion_matrix, log_loss

import matplotlib.pyplot as plt

from layer_model import CNN
model = CNN()
model.load_state_dict(torch.load('model_1_para.pth'))
# model.eval()

sampling_rate = 1000
totalscal = 128
wavename = 'cgau8'
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal  
scales = cparam / np.arange(totalscal, 0, -1)

folder_path = '../data/100ms_mix/group'


for j in range(1, 6, 1):
    output_list = []
    print(j)
    path = folder_path + str(j) + '/'
    data_names = os.listdir(path)
    print(len(data_names))
    for data_name in data_names:
        # print(data_name)
        file_path = path + data_name

        data = pd.read_csv(file_path, header=None, dtype=np.float32)
        signal = data[1]

        coefficients, frequencies = pywt.cwt(signal, scales, wavename, 1.0/1000)

        amp = abs(coefficients)
        amp = amp.reshape(1,1,128,1000)
        amp = torch.from_numpy(amp)
        # print(amp)
        # print(type(amp))
        # print(amp.shape)
        
        outputs = model(amp)
        output_max = outputs.argmax(dim=1)
        output_list.append(output_max.item())

    np.savetxt('predict_' + str(j) + '.csv', output_list, fmt='%d', delimiter=',')