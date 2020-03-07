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

# y_train = keras.utils.to_categorical(y_train)
# y_valid = keras.utils.to_categorical(y_valid)
# y_test = keras.utils.to_categorical(y_test)


#%%
# mnist take only 0 and 1

idx = np.where(np.logical_or((y_train_full == 0),(y_train_full == 1)))

y_train_full = y_train_full[idx]
X_train_full = np.squeeze(X_train_full[idx,:,:], axis=0)

idx2 = np.where(np.logical_or((y_test == 0),(y_test == 1)))

y_test = y_test[idx2]
X_test = np.squeeze(X_test[idx2,:,:], axis=0)

print(np.shape(y_test))
print(np.shape(X_test))

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)

#%%

curdir = '/home/yjsong/Git/Study/Study01/Chapter11'

model = keras.models.load_model(curdir+'/result/best_acc_model.hdf5')
model_clone = keras.models.clone_model(model)
model_clone.set_weights(model.get_weights())


#%%
model.summary()

# %%
model_clone.summary()

# %%
modelNew = keras.models.Sequential(model_clone.layers[:-1])
modelNew.add(keras.layers.Dense(1, activation='sigmoid', name='dense_4'))

for layer in modelNew.layers[:-1]:
    layer.trainable = False

modelNew.compile(loss='binary_crossentropy', 
            optimizer=keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy'])

# train few epochs with freeze
history = modelNew.fit(X_train, y_train, epochs=5,
                    validation_data=(X_valid, y_valid))

# unfreeze
for layer in modelNew.layers[:-1]:
    layer.trainable = True

# compile again (good idea to reduce the learning rate)
modelNew.compile(loss='binary_crossentropy', 
            optimizer=keras.optimizers.Adam(lr=1e-4),
            metrics=['accuracy'])

history = modelNew.fit(X_train, y_train, epochs=5,
                    validation_data=(X_valid, y_valid))


# %%

modelNew.evaluate(X_test, y_test)

# %%
