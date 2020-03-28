#%%
# generate data
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 2 patients, (10sample *2 lead)
X = tf.constant([[1.,3.,2.,6.,7.,9.,2.,3.,4.,3.,5.,2.,7.,9.,2.,3.,1.,1.,3.,9.],
                [14.,33.,52.,16.,72.,91.,12.,3.,54.,33.,15.,22.,37.,94.,22.,13.,11.,13.,35.,19.]])

X1 = tf.reshape(X[0,:], [2,10])
X2 = tf.reshape(X[1,:], [2,10])
print(X1)
print(X2)

meanX1 = tf.reduce_mean(X1, axis=1, keepdims=True)
stdX1 = tf.math.reduce_std(X1, axis=1, keepdims=True) 
meanX2 = tf.reduce_mean(X2, axis=1, keepdims=True)
stdX2 = tf.math.reduce_std(X2, axis=1, keepdims=True) 

X1New = tf.math.divide(tf.math.subtract(X1, meanX1), stdX1+keras.backend.epsilon())
print(X1New)
X2New = tf.math.divide(tf.math.subtract(X2, meanX2), stdX2+keras.backend.epsilon())
print(X2New)

#%%
# tf.data 에서 사용

dataset = tf.data.Dataset.from_tensor_slices(X)

for item in dataset:
    print(item)
print('\n')

def standardize_fn(x):

    xRe = tf.reshape(x, [2,10])

    xList = []
    for i in range(xRe.shape[0]):
        meanX = tf.reduce_mean(xRe[i,:])
        stdX = tf.math.reduce_std(xRe[i,:])
        
        sub = tf.math.subtract(xRe[i,:],meanX)
        xNew = tf.math.divide(sub,stdX+keras.backend.epsilon())
        xList.append(xNew)
        
    return tf.stack(xList)


dataset2 = dataset.map(standardize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
for item in dataset2:
    print(item)
print('\n')



#%%
# layer 에서 사용

dataset = tf.data.Dataset.from_tensor_slices(X)
dataset = dataset.batch(2)
for item in dataset:
    print(item)
print('\n')

class Standardization(keras.layers.Layer):

    @tf.function
    def call(self, inputs):
        
        xRe = tf.reshape(inputs, [-1,2,10])
        
        xList = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for j in tf.range(0, tf.shape(inputs)[0]):
            for i in range(xRe.shape[1]):
                meanX = tf.reduce_mean(xRe[j,i,:])
                stdX = tf.math.reduce_std(xRe[j,i,:])
            
                sub = tf.math.subtract(xRe[j,i,:],meanX)
                xNew = tf.math.divide(sub,stdX+keras.backend.epsilon())
                
                xList = xList.write(index=j*2+i, value=xNew)

        x = xList.stack()
        xx = tf.reshape(x, [-1,2,10])
        return xx


std_layer = Standardization()

input = keras.layers.Input(shape=[20])
output = std_layer(input)
model = keras.models.Model(inputs=input, outputs=output)

model.summary()

result = model.predict(dataset)
print(result)



#%%
# tf transform 으로 사용
import tensorflow_transform as tft
