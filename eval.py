import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np


datapath = './some_results/pred.mat'
data = scio.loadmat(datapath)

labelpath = './data/label.mat'
label_temp = scio.loadmat(labelpath)

result = data['result_all']
label = label_temp['label']

[m ,n] = label.shape

# '''image padding'''
# print(result[8,8], result[7,7])
# for i in range(8):
#     result[i, :] = result[8, :]
# for i in range(734, m):
#     result[i, :] = result[733, :]
# for j in range(8):
#     result[:, j] = result[:, 8]
# for j in range(1008, n):
#     result[:, j] = result[:, 1007]

pad_num = 0
for i in range(m):
    for j in range(n):
        if(result[i,j] == -1):
            pad_num += 1
print(pad_num)

'''image show'''

class_names = ['Stembeans', 'Peas', 'Forest', 'Lucerne','Wheat one', 'Beet', 'Potatoes', 'Bare soil', 
            'Grass', 'Rapeseed', 'Barley', 'Wheat two', 'Wheat three', 'Water', 'Buildings']
class_color = ['red', 'cyan', 'green', 'navyblue', 'yellow1', 'pink', 'Goldenrod1', 'Chocolate4', 'LimeGreen',
               'DeepPink4', 'Orange1', 'LightGreen', 'Thistle1', 'Blue1', 'Grey']
colors = [[255,0,0], [128,0,255], [0,128,0], [0,255,255], [255,128,255], 
          [128,0,128], [255,255,0], [128,128,0], [0,255,0], [255,128,0],
          [128,0,0], [128,128,255], [128,255,128], [0,0,255], [255,200,128]]

label_rgb = np.zeros((m,n,3), dtype=int)
result_rgb = np.zeros((m,n,3), dtype=int)

for i, color in zip(range(15), colors):
    label_rgb[(label - 1)== i] = color

for i, color in zip(range(1, 16), colors):
    result_rgb[(result + 1)== i] = color   

plt.axis('off')
plt.imshow(label_rgb)
plt.title('groundtruth')


plt.figure(2)
plt.axis('off')
plt.imshow(result_rgb)
plt.title('classification result')
plt.show() 


'''statistics'''

label_str = str(label.tolist())

clsnum_gt = np.zeros(16, dtype=int) # [0, 15]

clsnum_gt[0] = label_str.count('0') - label_str.count('10')
clsnum_gt[1] = label_str.count('1') - label_str.count('10') - 2 * label_str.count('11') - label_str.count('12') - label_str.count('13') - label_str.count('14') - label_str.count('15')
clsnum_gt[2] = label_str.count('2') - label_str.count('12')
clsnum_gt[3] = label_str.count('3') - label_str.count('13')
clsnum_gt[4] = label_str.count('4') - label_str.count('14')
clsnum_gt[5] = label_str.count('5') - label_str.count('15')
for i in range(6,16):         
    clsnum_gt[i] = label_str.count("{}".format(i))


label = label - 1
print(np.min(label), np.max(label))

result_num_correct = np.zeros(15, dtype=int) # [0, 14]
for i in range(m):
    for j in range(n):
        if(label[i,j] != 255):
            for k in range(15):
                if(label[i,j] == result[i,j] and label[i,j] == k):
                    result_num_correct[k] += 1

temp_label = 0
temp_test = 0
for i in range(1,16):
    print('accuracy of',i,':',result_num_correct[i-1]/clsnum_gt[i])
    temp_label += clsnum_gt[i]
    temp_test += result_num_correct[i-1]

print('total pixels:', temp_label)
print('total test pixels:', temp_test)
print('----overall accuracy:    ', temp_test/temp_label)    
    
print('----GroundTruth class nums----Test class nums----')
for i in range(1,16):
    print(i, ':', clsnum_gt[i],'     ',result_num_correct[i-1])



'''Confusion Matrix'''
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

# sns.set()
# f,ax = plt.subplots()

y_true, y_pred = [], []
for i in range(m):
    for j in range(n):
        if label[i,j] != 255:
            y_true.append(label[i,j])
            y_pred.append(result[i,j])
print(len(y_true), len(y_pred))
            
matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

diagonalSum = sum([matrix[i][i] for i in range(len(matrix))])
print('OA:  ', diagonalSum / np.sum(matrix))


po = diagonalSum / np.sum(matrix)
pe = sum([sum(matrix[i,:]) * sum(matrix[:,i]) for i in range(len(matrix))]) / np.sum(matrix) ** 2
k = (po - pe) / (1 - pe)
print('Kappa:', k)

kappa_value = cohen_kappa_score(y_true, y_pred)
print('Kappa:', kappa_value)
