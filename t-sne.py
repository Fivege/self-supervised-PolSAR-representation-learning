import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.manifold import TSNE
import seaborn as sns

from model_new import test_model
from data_get import x_6k, y_6k
import numpy as np

f_net = test_model()

x = tf.random.normal((512, 16, 16, 6))
h = f_net(x, training=False)
print('Initializing online networks...')
print('Shape of h:', h.shape)

# encoder_weights = 'pretrain.h5'
# f_net.load_weights(encoder_weights)
 
print('Weights of f loaded.')

"""
t-SNE visualization
"""
tsne = TSNE()

def plot_vecs_n_labels(v, labels, fname):
    fig = plt.figure(figsize = (8,6))
    fig.add_artist(patches.Rectangle((0.125, 1/12), 0.75, 5/6, linewidth = 1, edgecolor = 'black', facecolor = 'none'))
    plt.axis('off')
    # colors = [[255,0,0], [0,255,255], [0,255,0], [0,0,128], [238,238,0], 
    #         [255,192,203], [255,193,37], [139,69,19], [50,205,50], [139,10,80],
    #         [255,165,0], [144,238,144], [255,225,255], [0,0,255], [190,190,190]]
    
    colors = [[255,0,0], [128,0,255], [0,128,0], [0,255,255], [255,128,255], 
          [128,0,128], [255,255,0], [128,128,0], [0,255,0], [255,128,0],
          [128,0,0], [128,128,255], [128,255,128], [0,0,255], [255,200,128]]
    
    colors = [[float(j) / 255 for j in i] for i in colors]

    #sns.scatterplot(v[:,0], v[:,1], hue=labels, palette=colors, edgecolor='none')
    sns.scatterplot(v[:,0], v[:,1], hue=labels, palette=colors, edgecolor='none', legend=False)
    
    #sns.scatterplot(v[:,0], v[:,1], hue=labels, size=labels, sizes=(150,150),
    #                legend='brief', palette=colors, edgecolor='none')
                    # edgecolor='none' or linewidth=0
    
    #plt.legend(['Steambeans', 'Peas', 'Forest', 'Lucerne','Wheat one', 'Beet', 'Potatoes', 'Bare soil', 
                #'Grass', 'Rapeseed', 'Barley', 'Wheat two', 'Wheat three', 'Water', 'Buildings'], loc='best')
    plt.savefig(fname)



# change batch_size
# batch_size = 500
#data.shuffle_finetuning_data()
x, y = x_6k, np.squeeze(y_6k)
print(y.shape)
feature = f_net(x, training=False)
print(feature.shape)
pred_tsne = tsne.fit_transform(feature)
print(np.min(y), np.max(y))
print(pred_tsne.shape)


classes_name = ['Stembeans', 'Peas', 'Forest', 'Lucerne','Wheat one', 'Beet', 'Potatoes', 'Bare soil',
               'Grass', 'Rapeseed', 'Barley', 'Wheat two', 'Wheat three', 'Water', 'Buildings']
classes = []
pred_tsne_1 = np.zeros((y.shape[0], 2), dtype=float)
temp = 0
for j in range(15):
    for i in range(y.shape[0]):
        if y[i] == j:
            classes.append(classes_name[j])
            pred_tsne_1[temp] = pred_tsne[i]
            temp += 1

print(len(classes), pred_tsne_1.shape)

plot_vecs_n_labels(pred_tsne_1, classes, './some_results/random.png')