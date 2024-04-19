# MLSP
code for a paper

切分代码在slice.py

data文件夹中1,,5是原始数据，group1,,5是切分后的数据，json文件记录了切分的起始终止时间

num_list.npy记录了每一类有多少个数据csv，npy文件用np.load可以打开

filtered_1,,5是滤波后的数据，1/10/100ms_step基于这个切分
