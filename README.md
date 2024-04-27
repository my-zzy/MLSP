# MLSP
code for a paper

切分代码在slice.py

data文件夹中1,,5是原始数据，group1,,5是切分后的数据，json文件记录了切分的起始终止时间

num_list.npy记录了每一类有多少个数据csv，npy文件用np.load可以打开

filtered_1,,5是滤波后的数据，1/10/100ms_step基于这个切分

accuracy是正确率

Confusion_matrix是混淆矩阵

CrossEntropyLoss是每轮的交叉熵损失

predict1-5是对混合的预测，其中数字代表哪个动作参照

label_dict = dict({
    '静置': [1,0,0,0,0,0,0,0,0,0],	#静置是0，以此类推
    '咳嗽': [0,1,0,0,0,0,0,0,0,0],
    '哈欠': [0,0,1,0,0,0,0,0,0,0],
    '说话': [0,0,0,1,0,0,0,0,0,0],
    '干咽': [0,0,0,0,1,0,0,0,0,0],
    '提喉': [0,0,0,0,0,1,0,0,0,0],
    '喝水': [0,0,0,0,0,0,1,0,0,0],
    '咽水': [0,0,0,0,0,0,0,1,0,0],
    '咀嚼': [0,0,0,0,0,0,0,0,1,0],
    '咽食': [0,0,0,0,0,0,0,0,0,1]
})

code中：
layer_model.py是模型文件
slice和mix_cut是用来切割数据的
wvlt是小波变换
train_with_fig.py是训练代码

注意：如果想要运行代码，需要修改代码中所有的文件路径

为什么选择小波变换：可以把一维信号变成二维图片的样子，便于使用现有的网络结构进行训练。
为什么使用这样的网络：通过多个卷积层和池化层的堆叠，能够学习到从低级到高级的不同层次的特征表示（参照VGG的优点）。数据量不大，不足以支持更高级的网络。

训练集大小：9000
测试集大小：忘了，等会再说

小波变换选择的小波是：cgau8（忘了为什么选这个，好像是图片好看一点，如果要换也不复杂，跑下代码的时间）

batch_size=64，随手定的
epoch=10，十轮基本收敛了
