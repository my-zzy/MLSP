import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

with open("data/toenglish.json", 'r', encoding='utf-8') as f:
    toenglish = json.load(f)

with open("data/start_stop_time_1.json", 'r', encoding='utf-8') as f:
    start_stop_1 = json.load(f)

with open("data/start_stop_time_2.json", 'r', encoding='utf-8') as f:
    start_stop_2 = json.load(f)

with open("data/start_stop_time_3.json", 'r', encoding='utf-8') as f:
    start_stop_3 = json.load(f)

with open("data/start_stop_time_4.json", 'r', encoding='utf-8') as f:
    start_stop_4 = json.load(f)

with open("data/start_stop_time_5.json", 'r', encoding='utf-8') as f:
    start_stop_5 = json.load(f)

# five group of data
for k in range(1,6,1):
    folder_path = "data/filtered_{:d}".format(k)

    file_names = os.listdir(folder_path)

    # print(file_names)

    class_name = []
    num_list = []

    for i in file_names:
        if i[:2] == "混合":
            continue
        
        data = pd.read_csv(folder_path + "/" + i, header=None)
        x = data[0] * 1000                                      # convert to ms
        y = data[1]
        # plt.figure(figsize=(10, 6))
        # plt.plot(x, y)
        # plt.title(toenglish[i[:2]])
        # plt.xticks(np.arange(0, 12, 1))
        # plt.show()

        if k == 1:
            start_stop = start_stop_1
        elif k == 2:
            start_stop = start_stop_2
        elif k == 3:
            start_stop = start_stop_3
        elif k == 4:
            start_stop = start_stop_4
        elif k == 5:
            start_stop = start_stop_5
        else:
            print("out of bound")

        section = start_stop[i[:2]]
        print(i[:2] + " " + str(section))

        start = section[0] * 1000
        stop = section[1] * 1000
        step_len = 1                                         # step length
        width = 1000                                           # 1000ms width for one window
        num = int((stop - start - width) / step_len + 0.001)   # add 0.001 in case of floating point error
        print(num)
        class_name.append(str(i[:2]))
        num_list.append(str(num))
        cat = []
        for j in range(num):
            if j*step_len + width + start -1 > stop:
                print("out of range")
                print(j*step_len + width + start -1)
                print(stop)
                continue
            
            # if j > 500:
            #     break

            one_slice = np.array(y[int(j*step_len + start + 0.001): int(j*step_len + width + start + 0.001)])
            # print(one_slice.shape)    # 1000

            timestamp = np.arange(int(j*step_len + start + 0.001), int(j*step_len + width + start + 0.001))
            # print(timestamp.shape)    # 1000

            matrix_for_save = np.concatenate((timestamp.reshape(-1, 1), one_slice.reshape(-1, 1)), axis=1)
            # print(matrix_for_save)

            save_path = "data/{:d}ms_step/group{:d}/{:s}".format(step_len, k, i[:2])

            group_path = "data/{:d}ms_step/group{:d}".format(step_len, k)

            if not os.path.exists(group_path):
                os.mkdir(group_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            np.savetxt("data/{:d}ms_step/group{:d}/{:s}/{:d}.csv".format(step_len, k, i[:2], j), matrix_for_save, delimiter=",")

            # cat.append(matrix_for_save)

        # cat_array = np.concatenate(cat, axis=0)
        # np.savetxt("data/group2/{}.csv".format(i[:2]), cat_array, delimiter=",")

        # break
        
    class_name = np.array(class_name, dtype="str").reshape(-1,1)
    num_list = np.array(num_list, dtype="str").reshape(-1,1)
    np.save("data/{:d}ms_step/group{:d}/num_list".format(step_len, k), np.concatenate((class_name, num_list), axis=1))


