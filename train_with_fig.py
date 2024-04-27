import math
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, confusion_matrix, log_loss

import matplotlib.pyplot as plt

label_dict = dict({
    '静置': [1,0,0,0,0,0,0,0,0,0],
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

folder_path = '../data/10ms_cgau8/'

group_names = os.listdir(folder_path)

train_data = []
train_label = []
test_data = []
test_label = []

for group_name in group_names:
    print(group_name)
    group_path = folder_path + group_name + '/'
    data_names = os.listdir(group_path)
    for data_name in data_names:
        data_path = group_path + data_name
        file_names = os.listdir(data_path)
        for file_name in file_names:
            file_path = data_path + '/' + file_name
            data = pd.read_csv(file_path, header=None)
            if group_name == 'group5':
                test_data.append(np.array([data.values]))
                test_label.append(label_dict[data_name])
            else:
                train_data.append(np.array([data.values]))
                train_label.append(label_dict[data_name])
            # print(data.shape)
            # break

print('train_num: ', len(train_data))
print('test_num: ', len(test_data))

train_data = np.array(train_data)
train_label = np.array(train_label)
train_data = torch.tensor(train_data, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.float32)

test_data = np.array(test_data)
test_label = np.array(test_label)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_label = torch.tensor(test_label, dtype=torch.float32)

batch_size = 64
train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


from layer_model import CNN
model = CNN()

# from vgg import VGG16
# model = VGG16()

for inputs, labels in train_loader:

    print('inputs shape', inputs.shape)

    outputs = model(inputs)
    # print(outputs)
    break

# criterion = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_count = 0

print('start training')

total_correct = []
total_loss = []

for epoch in range(num_epochs):
    # print('epoch:', epoch)
    running_loss = 0.0
    
    model.train()
    for inputs, labels in train_loader:
        # print(inputs.shape)
        # print(inputs[0])
        # print(labels.shape)
        # print(labels[0])
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss
        running_loss += loss.item()

    # print(labels.size(0))
    # print(inputs.size(0))
    # print(outputs[:1])
    # print("loss one time ",loss.item())
    epoch_loss = running_loss / labels.size(0)
    total_loss.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # begin testing
    model.eval()
    total_count = 0

    count = 0
    num = 0
    y_true = []
    y_pred = []
    y_prob = []    

    with torch.no_grad():

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)

            # accuracy
            if outputs.shape == labels.shape:
                # print('outputs.shape', outputs.shape)
                output_max = outputs.argmax(dim=1)
                label_max = labels.argmax(dim=1)
                # print('output', output_max)
                # print('label', label_max)
                count = (output_max == label_max).sum().item()
                total_count += count

                y_true.extend(label_max.cpu().numpy())
                y_pred.extend(output_max.cpu().numpy())
                y_prob.extend(torch.softmax(outputs.cpu(), dim=1).numpy())
                num += labels.size(0)

            else:
                print("wrong shape")

        acc_rate = total_count / num
        total_correct.append(acc_rate)
        # break

        # print("count ", count)
        # print(labels.size(0))

        accuracy = count / num
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # print(0)
        # print(y_true)
        # print(y_pred)
        # print(y_prob)

    
        # start drawing
        if epoch+1 == num_epochs:
            # 绘制准确率曲线
            plt.figure()
            plt.plot(total_correct)
            print(1)
            print(total_correct)
            plt.title('Accuracy')
            plt.xticks(np.arange(num_epochs), np.arange(num_epochs))
            plt.yticks(np.arange(0,1.2,0.2), ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            for i, y_val in enumerate(total_correct):
                plt.annotate(f'{y_val:.2f}', (i, total_correct[i]), textcoords="offset points", xytext=(0,10), ha='center')

            plt.savefig('accuracy_curve_' + str(num_epochs) + '.png')

            # # 绘制精确率-召回率曲线
            # precision, recall, _ = precision_recall_curve(y_true, y_prob)
            # plt.figure()
            # plt.plot(recall, precision, marker='.')
            # plt.title('Precision-Recall Curve')
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.savefig('precision_recall_curve.png')

            # 绘制混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(2)
            print(cm_normalized)
            print(cm)
            # print(type(cm))
            # rounded_cm = np.round(cm_normalized, decimals=1)
            rounded_cm = np.round(cm, decimals=1)
            plt.figure()
            plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            num_classes = 10
            plt.xticks(np.arange(num_classes), np.arange(num_classes))
            plt.yticks(np.arange(num_classes), np.arange(num_classes))
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            plt.colorbar()
            # 显示数值
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, rounded_cm[i, j], ha='center', va='center')
            plt.savefig('confusion_matrix_' + str(num_epochs) + '.png')

            # 绘制对数损失曲线
            plt.figure()
            plt.plot(total_loss)
            plt.title('CrossEntropyLoss')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            print(3)
            print(total_loss)
            for i, y_val in enumerate(total_loss):
                plt.annotate(f'{y_val:.2f}', (i, total_loss[i]), textcoords="offset points", xytext=(0,10), ha='center')

            plt.savefig('log_loss_curve_' + str(num_epochs) + '.png')

            torch.save(model, 'model_1.pth')
            torch.save(model.state_dict(), 'model_1_para.pth')
            np.savetxt('Accuracy.csv', total_correct, fmt='%.6f', delimiter=',')
            np.savetxt('Confusion_matrix.csv', cm, fmt='%d', delimiter=',')
            np.savetxt('CrossEntropyLoss.csv', total_loss, fmt='%.6f', delimiter=',')
        

        # total_correct_ave = sum(total_correct) / len(total_correct)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Acc Rate: {acc_rate}")

        





        # if total_correct > best_count:
        #     torch.save(model.state_dict(), 'model_1.pth')