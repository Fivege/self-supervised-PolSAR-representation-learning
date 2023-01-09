import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


row = 750
col = 1024
channels = 6 
datafile ='./data/data.mat'
labelfile = './data/label.mat'
data_temp = scio.loadmat(datafile)
label_temp = scio.loadmat(labelfile)
labels = label_temp['label']

imgs = np.zeros((row,col,channels), dtype=np.float32) 
imgs[:,:,0] = data_temp['A']
imgs[:,:,1] = data_temp['B']
imgs[:,:,2] = data_temp['C']
imgs[:,:,3] = data_temp['D']
imgs[:,:,4] = data_temp['E']
imgs[:,:,5] = data_temp['F']


n = 16
m = 8
print(imgs[100-m :100+m, 100-m :100+m, :].shape)

'''Train samples: every classes randomly select 400'''
nums = 6000  #number of train samples
num_classes = 15

labels = labels - 1  #[0, 14]
each_cls_num = 400
cls_num = np.zeros((15), dtype=int)
x_patch = np.zeros((15, each_cls_num, n, n, channels), dtype=np.float32)
y_patch = -np.ones((15, each_cls_num, 1), dtype=int)


'''set samples selected radio for each class'''
radio = [0.067, 0.045, 0.029, 0.044, 0.025, 0.041, 0.027, 0.135, 0.065, 0.033, 0.057, 0.039, 0.019, 0.031]
for i in range(row):
    for j in range(col):
        if labels[i,j] == 14 and random.random() < 0.9:
            if (i-m)>0 and (i+m)<row and (j-m)>0 and (j+m)<col and cls_num[14] < each_cls_num:
                x_patch[14, cls_num[14], :, :, :] = imgs[i-m :i+m, j-m :j+m, :]
                y_patch[14, cls_num[14]] = labels[i,j]
                cls_num[14] += 1    
        for k in range(14):
            if (i-m)>0 and (i+m)<row and (j-m)>0 and (j+m)<col and cls_num[k] < each_cls_num:
                if labels[i,j] == k and random.random() < radio[k] :
                    x_patch[k, cls_num[k], :, :, :] = imgs[i-m :i+m, j-m :j+m, :]
                    y_patch[k, cls_num[k]] = labels[i,j]
                    cls_num[k] += 1
# x_eval = x_patch.reshape(-1,n,n,channels)
# y_eval = y_patch.reshape(-1,1)

# print([cls_num[k] for k in range(15)])
# x_train = x_patch.reshape(-1,n,n,channels)  #维度合并
# y_train = y_patch.reshape(-1,1)
# print([np.min(y_train), np.max(y_train)])
# print(x_train.shape, (y_train.shape))


# train_data = list(zip(x_train, y_train)) # Y: 0 ~ 14
# random.shuffle(train_data)
# x_train[:,:,:,:], y_train[:] = zip(*train_data) #train data
# plt.imshow(x_train[0,:,:,0:3]) #[i : j] 
# plt.show()

x_6k = x_patch.reshape(-1,n,n,channels)
y_6k = y_patch.reshape(-1,1)

print([cls_num[k] for k in range(15)])

print([np.min(y_6k), np.max(y_6k)])
print(x_6k.shape, (y_6k.shape))


train_data = list(zip(x_6k, y_6k)) # Y: 0 ~ 14
random.shuffle(train_data)
x_6k[:,:,:,:], y_6k[:] = zip(*train_data)


'''images for linear evaluation(20 or 300)'''
x_eval = np.zeros((15, 20, n, n, channels), dtype=np.float32)
y_eval = -np.ones((15, 20, 1), dtype=int)
eval_num = np.zeros((15), dtype=int)
for i in range(each_cls_num):
    for k in range(15):
        if random.random() <= 0.08 and eval_num[k] < 20:
            x_eval[k, eval_num[k], :, :, :] = x_patch[k, i, :, :, :]
            y_eval[k, eval_num[k]] = y_patch[k, i]
            eval_num[k] += 1
   
print([eval_num[k] for k in range(15)])
x_eval = x_eval.reshape(-1,n,n,channels)
y_eval = y_eval.reshape(-1,1)
print([np.min(y_eval), np.max(y_eval)])
print('eval:', x_eval.shape, (y_eval.shape))


'''images for unsupervised pretraining (don't use label y_train)'''
temp = 0
train_num = 40000
x_train = np.zeros((train_num, n, n, channels), dtype=np.float32)
y_train = - np.ones((train_num, 1), dtype=int)
for i in range(row):
    for j in range(col):                             
        if (i-m)>0 and (i+m)<row and (j-m)>0 and (j+m)<col and temp < train_num and random.random() < 0.06:
            x_train[temp, :, :, :] = imgs[i-m :i+m, j-m :j+m, :]   
            y_train[temp, :] = labels[i,j]
            temp += 1

train_data = list(zip(x_train, y_train)) # Y: 0 ~ 14
random.shuffle(train_data)
x_train[:,:,:,:], y_train[:] = zip(*train_data) #train data
print('pretrain:', [np.min(y_train), np.max(y_train)])
print(x_train.shape, (y_train.shape))
print(temp)


'''images for test'''
temp_num = 0
test_num = 155500
X_all = np.zeros((test_num, n, n, channels), dtype=np.float32)
Y_all = - np.ones((test_num, 1) ,dtype=int)
local_all = np.zeros((test_num,2), dtype=int)
for i in range(3, row-3):
    for j in range(3, col-3):
        if labels[i][j] == 14 :
            if (i-m)>0 and (i+m)<row and (j-m)>0 and (j+m)<col and temp_num < test_num:
                X_all[temp_num, :, :, :] = imgs[i-m :i+m, j-m :j+m, :]
                Y_all[temp_num] = labels[i,j]
                local_all[temp_num,:] =[i,j] 
                temp_num = temp_num + 1
        elif labels[i][j] != 255 :
            if (i-m)>0 and (i+m)<row and (j-m)>0 and (j+m)<col and temp_num < test_num:

                X_all[temp_num, :, :, :] = imgs[i-m :i+m, j-m :j+m, :]
                Y_all[temp_num] = labels[i,j]
                local_all[temp_num,:] =[i,j] 
                temp_num = temp_num + 1

print('test:', [np.min(Y_all), np.max(Y_all)])
print([np.min(labels), np.max(labels)], labels.dtype)
print(temp_num)




padimage = np.zeros((row + 16, col + 16, channels), dtype=np.float32)
padimage[8:758, 8:1032, :] = imgs[:, :, :]
padimage[754:758, 8:1032, :] = padimage[750:754, 8:1032, :]
for i in range(8):
    padimage[i, 8:1032, :] = padimage[15-i, 8:1032, :]
for i in range(758, 766):
    padimage[i, 8:1032, :] = padimage[1515-i, 8:1032, :]
for j in range(8):
    padimage[:, j, :] = padimage[:, 15-j, :]
for j in range(1032, 1040):
    padimage[:, j, :] = padimage[:, 2063-j, :]

temp_num = 0
all_num = 750 * 1024
x_pred = np.zeros((all_num, n, n, channels), dtype=np.float32)
y_pred = - np.ones((all_num, 1) ,dtype=int)
local_pred = np.zeros((all_num,2), dtype=int)

row = row + 16
col = col + 16
for i in range(row):
    for j in range(col):
        if (i-m)>=0 and (i+m)<row and (j-m)>=0 and (j+m)<col and temp_num < all_num:

            x_pred[temp_num, :, :, :] = padimage[i-m :i+m, j-m :j+m, :]
            y_pred[temp_num] = labels[i-8,j-8]
            local_pred[temp_num,:] =[i-8,j-8]
            temp_num = temp_num + 1
print(temp_num)