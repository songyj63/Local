#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image


# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255

images = np.array([china, flower])
batch_size, height, width, channels = images.shape

print(images.shape)

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # vertical line
filters[3, :, :, 1] = 1 # horizontal line
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
plt.show()


# layer method
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                            padding='same', activation='relu')


# pooling
max_poll = keras.layers.MaxPool2D(pool_size=2, padding='same')

#%%
# depth wise max pooling

# layer example
depth_pool = keras.layers.Lambda(
                lambda X: tf.nn.max_pool(X, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3),
                                        padding="valid"))

# implementation test
print(china.shape)
print(china[0,0,0])
print(china[0,0,1])
print(china[0,0,2])

x = tf.nn.max_pool(images, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3),
                                        padding='VALID')

print(x.shape)
print(x[0,0,0])

# %%
# gloval average pooling
global_av_pool = keras.layers.GlobalAvgPool2D()

# %%
# resnet-34

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,    
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]
        
        self.skip_layers = []
        
        if strides > 1:
            self.skip_layers = [
            keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization()
        ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

#%%
from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
images = np.array([china, flower])

plt.imshow(china)
plt.show()
plt.imshow(flower)
plt.show()

# pretrained model from keras
model = keras.applications.resnet50.ResNet50(weights='imagenet')

# resNet-50 model expects 224x224 image size
images_resized = tf.image.resize(images, [224, 224])

# preprocess image
inputs = keras.applications.resnet50.preprocess_input(images_resized)

# output 1000 classes
y_proba = model.predict(inputs)
print(y_proba)
print(y_proba.shape)

top_K = keras.applications.resnet50.decode_predictions(y_proba, top=3)
for image_index in range(len(images)):
    print("Image #{}".format(image_index))
    for class_id, name, y_proba in top_K[image_index]:
        print(" {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
    print()