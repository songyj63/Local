#%%
# load data
import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


dataset = keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = dataset

print(np.shape(X_train_full))
print(np.shape(X_test))

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)

y_train = keras.utils.to_categorical(y_train)
y_valid = keras.utils.to_categorical(y_valid)
y_test = keras.utils.to_categorical(y_test)

# %%
# build model

input = keras.layers.Input(shape=[28,28])
flatten = keras.layers.Flatten()(input)
hidden1 = keras.layers.Dense(300, activation='relu')(flatten)
hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)
hidden3 = keras.layers.Dense(50, activation='relu')(hidden2)
output = keras.layers.Dense(10, activation='softmax')(hidden3)
model = keras.Model(inputs=[input], outputs=[output])

model.summary()

model.compile(loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])

# train
curdir = '/home/yjsong/Git/Study/Study01/Chapter10'
root_logdir = os.path.join(curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# best model
# filepath_loss=os.path.join(curdir, "result/best_loss_model.hdf5")
filepath_acc=os.path.join(curdir, "result/best_acc_model.hdf5")

# if model exist
# model = keras.models.load_model(filepath) # roll back to best model

# checkpoint_cb_loss = keras.callbacks.ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True)
checkpoint_cb_acc = keras.callbacks.ModelCheckpoint(filepath_acc, monitor='val_accuracy', verbose=1, save_best_only=True)
# early stopping option
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
# tensor board
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb_acc, early_stopping_cb, tensorboard_cb])

# tensorboard --logdir=/home/yjsong/Git/Study/Study01/Chapter10/my_logs --port=6006
# local anaconda: ssh -NL 6006:localhost:6006 yjsong@210.116.109.43
# or
# terminal에 아래 적고 http://localhost:6006 접속
# tensorboard --logdir=C:\Users\VUNO\Desktop\Programming\Git\Study\Study01\Chapter10\my_logs --port=6006

#%%

# filepath_loss=os.path.join(curdir, "result/best_loss_model.hdf5")
filepath_acc=os.path.join(curdir, "result/best_acc_model.hdf5")

# model_loss = keras.models.load_model(filepath_loss) # roll back to best model
model_acc = keras.models.load_model(filepath_acc) # roll back to best model

# model_loss.evaluate(X_test, y_test)
print(model_acc.evaluate(X_test, y_test))


#%%
# # fine-tuning model, searching hyperparameters

# def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28,28]):
#     model = keras.models.Sequential()
#     model.add(keras.layers.InputLayer(input_shape=input_shape))
#     model.add(keras.layers.Flatten())
#     for layer in range(n_hidden):
#         model.add(keras.layers.Dense(n_neurons, activation='relu'))
#     model.add(keras.layers.Dense(10, activation='softmax'))
#     optimizer = keras.optimizers.Adam(lr=learning_rate)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     model.summary()
#     return model

# keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# curdir = '/home/yjsong/Git/Study/Study01/Chapter10'
# root_logdir = os.path.join(curdir, "my_logs")

# def get_run_logdir():
#     import time
#     run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
#     return os.path.join(root_logdir, run_id)

# run_logdir = get_run_logdir()

# # best model
# filepath=os.path.join(curdir, "result/best_model.hdf5")

# checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True)
# # early stopping option
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
# # tensor board
# tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# # keras_reg.fit(X_train, y_train, epochs=100,
# #                 validation_data=(X_valid, y_valid),
# #                 callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

# # mse_test = keras_reg.score(X_test, y_test)

# #%%


# param_distribs = {
#     "n_hidden": [0, 1, 2, 3],
#     "n_neurons": np.arange(1, 100),
#     "learning_rate": reciprocal(3e-4, 3e-2),
# }

# rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, verbose=1, n_iter=2, cv=2)
# rnd_search_cv.fit(X_train, y_train, epochs=10,
#                     validation_data=(X_valid, y_valid),
#                     callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

# print(rnd_search_cv.best_params_)
# print(rnd_search_cv.best_score_)
# model = rnd_search_cv.best_estimator_.model

# model.save("my_best_model.h5")

# %%
