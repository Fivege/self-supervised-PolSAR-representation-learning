import tensorflow as tf
from tensorflow import keras
from keras import layers
from condconv import CondConv2D
    
class test_model(keras.Model):
    
    def __init__(self, filter_num=32, number_experts=3):
        super(test_model, self).__init__()
        
        self.conv_1 = CondConv2D(32, kernel_size=3, stride=1, use_bias=False, num_experts=number_experts, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.relu_1 = layers.Activation('relu')
        self.pool_1 = layers.MaxPool2D((2,2), strides=2, padding='same')
        
        self.conv_2 = CondConv2D(64, kernel_size=3, stride=1, use_bias=False, num_experts=number_experts, padding='same')
        self.bn_2 = layers.BatchNormalization()
        self.relu_2 = layers.Activation('relu')
        self.pool_2 = layers.MaxPool2D((2,2), strides=2, padding='same')
        
        self.conv_3 = CondConv2D(128, kernel_size=3, stride=1, use_bias=False, num_experts=number_experts, padding='same')
        self.bn_3 = layers.BatchNormalization()
        self.avgpool = layers.GlobalAveragePooling2D() #(1,1,channel)
        
        #SE block
        self.se_globalpool = layers.GlobalAveragePooling2D()
        self.se_resize = layers.Reshape((1,1,filter_num))
        self.se_fc1 = layers.Dense(units=filter_num // 16, activation='relu', use_bias=False)
        self.se_fc2 = layers.Dense(units=filter_num, activation='sigmoid', use_bias=False)
        
        #SE block
        self.se_resize_2 = layers.Reshape((1,1,64))
        self.se_fc1_2 = layers.Dense(units=64 // 16, activation='relu', use_bias=False)
        self.se_fc2_2 = layers.Dense(units=64, activation='sigmoid', use_bias=False)
        
    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = self.relu_1(x)
        x = self.pool_1(x)
        
         #SE block
        b = x
        out = self.se_globalpool(x)
        out = self.se_resize(out)
        out = self.se_fc1(out)
        out = self.se_fc2(out)
        x = layers.Multiply()([b, out])
        
        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = self.relu_2(x)
        x = self.pool_2(x)
        
         #SE block
        b = x
        out = self.se_globalpool(x)
        out = self.se_resize_2(out)
        out = self.se_fc1_2(out)
        out = self.se_fc2_2(out)
        x = layers.Multiply()([b, out])
       
        x = self.conv_3(x)
        x = self.bn_3(x, training=training)
        out = self.avgpool(x)

        return out



class ProjectionHead(tf.keras.Model):

    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=256)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=128)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

class Prediction(tf.keras.Model):
    
    def __init__(self):
        super(Prediction, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=256)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=128)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x


class ClassificationHead(tf.keras.Model):

    def __init__(self):
        super(ClassificationHead, self).__init__()
        
        self.fc = tf.keras.layers.Dense(units=15)

    def call(self, inp):
        x = self.fc(inp)
        
        return x