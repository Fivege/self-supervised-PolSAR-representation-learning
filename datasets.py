import tensorflow as tf
import numpy as np

from augmentation import augment_image_pretraining, augment_image_finetuning, augment_image_pretraining_mixup
from data_get import x_train, y_train, X_all, Y_all, x_eval, y_eval, x_pred, y_pred
 

def two_image_mix(x1, x2, alpha):
    
    lam = np.random.beta(alpha, alpha, size=[])
    lam = tf.cast(lam, dtype=tf.float32)
    
    
    output = x1 * lam  +  x2 * (1. - lam)
    return output

class polsar:
    
    def __init__(self):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = X_all, Y_all
        self.x_eval, self.y_eval = x_eval, y_eval
        self.x_pred, self.y_pred = x_pred, y_pred
        
        self.num_train_images, self.num_test_images = self.y_train.shape[0], self.y_test.shape[0]
        self.num_eval_images = self.y_eval.shape[0]
        self.num_all = self.y_pred.shape[0]
        self.class_names = ['Buildings', 'Rapeseed', 'Beet', 'Steambeans', 'Peas', 'Forest', 'Lucerne', 'Potatoes'
                            'Bare soil', 'Grass', 'Barley', 'Water', 'Wheat one', 'Wheat two', 'Wheat three']
        
        self.x_train = tf.image.convert_image_dtype(self.x_train, tf.float32)
        self.x_test = tf.image.convert_image_dtype(self.x_test, tf.float32)
        self.x_eval = tf.image.convert_image_dtype(self.x_eval, tf.float32)
        self.x_pred = tf.image.convert_image_dtype(self.x_pred, tf.float32)

        self.y_train = tf.cast(tf.squeeze(self.y_train), tf.int32)
        self.y_test = tf.cast(tf.squeeze(self.y_test), tf.int32)
        self.y_eval= tf.cast(tf.squeeze(self.y_eval), tf.int32)
        self.y_pred= tf.cast(tf.squeeze(self.y_pred), tf.int32)
    
    def get_batch_pretraining(self, batch_id, batch_size):
        
        thisbatch = self.x_train[batch_id*batch_size:(batch_id+1)*batch_size]
        
        augmented_images_1, augmented_images_2 = [], []
        mix_image = []
        
        for image_id in range(batch_size):
            image = thisbatch[image_id]
            
            image1 = augment_image_pretraining_mixup(image)
            image2 = augment_image_pretraining_mixup(image)
            mix = two_image_mix(image1, image2, 1.0)
            
            # augmented_images_1.append(augment_image_pretraining_mixup(image1))
            # augmented_images_2.append(augment_image_pretraining_mixup(image2))
            augmented_images_1.append(image1)
            augmented_images_2.append(image2)
            mix_image.append(mix)
            
        x_batch_1 = tf.stack(augmented_images_1)
        x_batch_2 = tf.stack(augmented_images_2)
        mix_batch = tf.stack(mix_image)
        
        
        return x_batch_1, x_batch_2, mix_batch    # (bs, 16, 16, 6), (bs, 16, 16, 6)
    
    
    def get_batch_finetuning(self, batch_id, batch_size):
        augmented_images = []
        for image_id in range(batch_id*batch_size, (batch_id+1)*batch_size):
            image = self.x_eval[image_id]
            augmented_images.append(augment_image_finetuning(image))
        
        x_batch = tf.stack(augmented_images)
        y_batch = tf.slice(self.y_eval, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)


    def get_batch_testing(self, batch_id, batch_size):
        x_batch = tf.slice(self.x_test, [batch_id*batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        y_batch = tf.slice(self.y_test, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)
    
    def get_batch_predict(self, batch_id, batch_size):
        x_batch = tf.slice(self.x_pred, [batch_id*batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        y_batch = tf.slice(self.y_pred, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  


    def shuffle_training_data(self):
        random_ids = tf.random.shuffle(tf.range(self.num_train_images))
        self.x_train = tf.gather(self.x_train, random_ids)
        self.y_train = tf.gather(self.y_train, random_ids)
        
    def shuffle_finetuning_data(self):
        random_ids = tf.random.shuffle(tf.range(self.num_eval_images))
        self.x_eval = tf.gather(self.x_eval, random_ids)
        self.y_eval = tf.gather(self.y_eval, random_ids)

