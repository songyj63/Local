#%%
# load data
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


#%%
# split data
print(X_train_full.shape)
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# %%
# build model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# # could do in this way
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])

model.summary()

print(model.layers)
hidden1 = model.layers[1]
print(hidden1.name)

# %%
# check model
weights, biases = hidden1.get_weights()
print(weights)
print(biases)
print(biases.shape)

# %%
# initialize model

model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=0.01),
                metrics=["accuracy"])

# sparse_categorical_crossentropy: label from 0 to 9
# categorical_crossentropy: one-hot vectors [0, 0, 1]
# binary classification:
# binary_crossentropy with sigmoid activation at the end

# %%
# train

history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid))

# history.params: training parameters
# history.epoch: the list of epochs
# history.history: **important** dictionary, loss and extra metrics it measured at the end of each epoch on the training

# train continue: fit() method again, 
# since Keras just continues training where it left off

# %%
# draw training history

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

# %%
# testing

model.evaluate(X_test, y_test)
# %%
# prediction with new data (probability)
X_new = X_test[:5]
y_proba = model.predict(X_new)
print(y_proba.round(2))

# prediction (class)
y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])

# %%
# model.save("my_keras_model.h5")

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model_1.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=3, callbacks=[checkpoint_cb])
# %%

print(history.history)

# %%
