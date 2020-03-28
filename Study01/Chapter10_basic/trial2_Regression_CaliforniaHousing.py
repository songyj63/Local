#%%
# load data
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = np.loadtxt('C:/Users/VUNO/Desktop/Programming/Git/Study/Study01/Chapter10/data/housing.data.txt')
X = data[:,0:13]
Y = data[:,13]

X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
#     keras.layers.Dense(1)
# ])


# input_ = keras.layers.Input(shape=X_train.shape[1:])
# hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.Concatenate()([input_, hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.Model(inputs=[input_], outputs=[output])

# model.summary()
# model.compile(loss="mean_squared_error", 
#             optimizer=keras.optimizers.SGD(lr=0.01))
# history = model.fit(X_train, y_train, epochs=100,
#                     validation_data=(X_valid, y_valid))

# mse_test = model.evaluate(X_test, y_test)
# X_new = X_test[:3] # pretend these are new instances
# y_pred = model.predict(X_new)

# print(y_pred)
# print(Y[:3])
# %%
# # multi input model
# input_A = keras.layers.Input(shape=[6], name="wide_input")
# input_B = keras.layers.Input(shape=[7], name="deep_input")
# hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.concatenate([input_A, hidden2])
# output = keras.layers.Dense(1, name="output")(concat)
# model = keras.Model(inputs=[input_A, input_B], outputs=[output])

# model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

# X_train_A, X_train_B = X_train[:, :6], X_train[:, 6:]
# X_valid_A, X_valid_B = X_valid[:, :6], X_valid[:, 6:]
# X_test_A, X_test_B = X_test[:, :6], X_test[:, 6:]
# X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
# validation_data=((X_valid_A, X_valid_B), y_valid))

# mse_test = model.evaluate((X_test_A, X_test_B), y_test)

# y_pred = model.predict((X_new_A, X_new_B))

# print(y_pred)
# print(Y[:3])
# %%
# multi output model

# input_A = keras.layers.Input(shape=[6], name="wide_input")
# input_B = keras.layers.Input(shape=[7], name="deep_input")
# hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.concatenate([input_A, hidden2])
# output = keras.layers.Dense(1, name="output")(concat)
# output = keras.layers.Dense(1, name="main_output")(concat)
# aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)

# model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

# model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))

# X_train_A, X_train_B = X_train[:, :6], X_train[:, 6:]
# X_valid_A, X_valid_B = X_valid[:, :6], X_valid[:, 6:]
# X_test_A, X_test_B = X_test[:, :6], X_test[:, 6:]
# X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
#                     validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))

# total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])

# y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

# print(y_pred_main)
# print(y_pred_aux)
# print(Y[:3])

#%%

# save & load best model

input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.summary()

model.compile(loss="mean_squared_error", 
            optimizer=keras.optimizers.SGD(lr=0.01))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])


model = keras.models.load_model("my_keras_model.h5") # roll back to best model

mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3] # pretend these are new instances
y_pred = model.predict(X_new)

print(y_pred)
print(Y[:3])


# %%

import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

model.compile(loss="mean_squared_error", 
                optimizer=keras.optimizers.SGD(lr=0.01))

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])

# %%
