#%%
# leakyReLU, selu
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dens(10, kernel_initializer='he_normal'),
    keras.layers.LeakyReLU(alpha=0.2),

    keras.layers.Dense(10, activation='selu', kernel_initializer='lecun_nrmal')
])


#%%
# batch norm eaxmple

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax")
])

#%%
# learning rate scheduling

# 1. Power scheduling method
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)


# 2. Exponential scheduling method
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])

# 3. piecewise method / epoch 마다 특정 값으로 줄임

def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001

# 위 방법들은 epoch 마다 변하는 것들이라. model load 시에는 epoch가 0으로 리셋되는 것
# 방지 위해 initial_epoch argument를 fit에서 활용해야 함.


# 4. performance scheduling

# lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# factor: new_lr = lr * factor
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5)


# %%
# 가중치 감소

# l2 norm, l1 norm 도 마찬가지로
# 둘다 적용하고싶으면 keras.regularizers.l1_l2()
layer = keras.layers.Dense(100, activation="elu",
    kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2(0.01))


#%%
# dropout

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])


