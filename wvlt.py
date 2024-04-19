import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import os
 
# 生成三个不同频率成分的信号  3000个点
# fs = 1000  # 采样率
# time = np.linspace(0, 1, fs, endpoint=False)  # 时间
# # 第一个频率成分
# signal1 = np.sin(2 * np.pi * 30 * time)
# # 第二个频率成分
# signal2 = np.sin(2 * np.pi * 60 * time)
# # 第三个频率成分
# signal3 = np.sin(2 * np.pi * 120 * time)
# # 合并三个信号
# signal = np.concatenate((signal1, signal2, signal3))



# 采样频率
sampling_rate = 1000
 
# 尺度长度
totalscal = 128

# 小波基函数
wavename = 'cgau8'
# wavename = 'morl'
 
# 小波函数中心频率
fc = pywt.central_frequency(wavename)
 
# 常数c
cparam = 2 * fc * totalscal  

# 尺度序列
scales = cparam / np.arange(totalscal, 0, -1)

for k in range(1,6,1):
    folder_path = 'data/100ms_step/group' + str(k)
    file_names = os.listdir(folder_path)
    for i in file_names:
        if i == 'num_list.npy':
            continue
        
        file_namess = os.listdir(folder_path + '/' + i)
        for j in file_namess:
            data = pd.read_csv(folder_path + '/' + i + '/' + j, header=None)
            signal = data[1]

            # 进行CWT连续小波变换
            coefficients, frequencies = pywt.cwt(signal, scales, wavename, 1.0/1000)
            
            # 小波系数矩阵绝对值
            amp = abs(coefficients)
            print(type(amp))
            print(amp.shape)
            save_path = 'data/test_100ms_cgau8_6f/group' + str(k)
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            save_path = save_path + '/' + i
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            np.savetxt(save_path + '/' + j[:-4] + '.csv', amp, fmt='%.6f', delimiter=',')
            print(np.max(amp))
            
            

            # 根据采样频率 sampling_period 生成时间轴 t
            # t = np.linspace(0, 1.0/sampling_rate, sampling_rate, endpoint=False)

            # 绘制时频图谱
            # plt.figure(figsize=(10,10))
            # plt.subplot(2,1,1)
            # plt.plot(signal)
            # plt.title('filtered original signal')

            # plt.subplot(2,1,2)
            # plt.contourf(t, frequencies, amp, cmap='jet')
            # plt.title('CWT')
            # plt.legend().remove()
            # plt.show()

# 读取信号数据
# folder_path = 'data/100ms_step/group1/干咽'
# file_names = os.listdir(folder_path)
# for i in file_names:

#     data = pd.read_csv(folder_path + '/' + i, header=None)
#     signal = data[1]

#     # 进行CWT连续小波变换
#     coefficients, frequencies = pywt.cwt(signal, scales, wavename, 1.0/1000)
    
#     # 小波系数矩阵绝对值
#     amp = abs(coefficients)

#     # 根据采样频率 sampling_period 生成时间轴 t
#     t = np.linspace(0, 1.0/sampling_rate, sampling_rate, endpoint=False)

#     # 绘制时频图谱
#     plt.figure(figsize=(10,6))
#     plt.subplot(2,1,1)
#     plt.plot(signal)
#     plt.title('filtered original signal')

#     plt.subplot(2,1,2)
#     plt.contourf(t, frequencies, amp, cmap='jet')
#     plt.title('CWT')
#     plt.show()