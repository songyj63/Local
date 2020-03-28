#%%
import tensorflow as tf

t = tf.constant([[1.,2.,3.], [4.,5.,6.]])
print(t)
print(t.shape)
print(t.dtype)

print(t[:,1:])

print(t[:,1])
print(t[:,1, tf.newaxis])

print(tf.square(t))

print(tf.transpose(t))
#%%
# tensor and numpy
import numpy as np
import tensorflow as tf

a = np.array([2., 4., 5.])
t = tf.constant(a)
print(t)

print(t.numpy())

print(tf.square(a))
print(np.square(t))

#%%
# type conversion

# error (tensor default float32)
# tf.constant(2.) + tf.constant(40)
# tf.constant(2.) + tf.constant(40., dtype=tf.float64)

t2 = tf.constant(40., dtype=tf.float64)
t = tf.constant(2.0) + tf.cast(t2, tf.float32)
print(t)


#%%
# variables

v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
v.assign(2*v)
print(v)
v[0, 1].assign(42)
print(v)
v[:, 2].assign([0., 1.])
print(v)

v.scatter_nd_update(indices=[[0,0], [1,2]], updates=[100., 200.])
print(v)

#%%
class AdvancedList(list):
    
    def replace(self, old, new):
        while old in self:
            self[self.index(old)] = new
                    
                                                                
x = AdvancedList([1, 2, 3, 1, 2, 3, 1, 2, 3])
x.replace(1, 100)
print(x)


#%%
# custom loss

# method 1. function 
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])

model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})

# method 2. 
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")

model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})


# method 3. class
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

model.compile(loss=HuberLoss(2.), optimizer="nadam")

model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
                                custom_objects={"HuberLoss": HuberLoss})

#%%
# custom activation function

def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)
def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)
def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))
def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


layer = keras.layers.Dense(30, activation=my_softplus,
                            kernel_initializer=my_glorot_initializer,
                            kernel_regularizer=my_l1_regularizer,
                            kernel_constraint=my_positive_weights)



#%%
# custom training loop

# build simple model
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal", kernel_regularizer=l2_reg),
    keras.layers.Dense(1, kernel_regularizer=l2_reg)
])

# random sampling fn
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

# print status
def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)


# define hyperparameters
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]


# custom loop
for epoch in range(1, n_epochs+1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps+1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()



