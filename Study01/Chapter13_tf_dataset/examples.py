#%%
# dataset

import tensorflow as tf
from tensorflow import keras

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)

print(X)

for item in dataset:
    print(item)

# batch, repeat
dataset2 = dataset.repeat(3).batch(7)
for item in dataset2:
    print(item)

# map
dataset3 = dataset.map(lambda x: x*2)
dataset3 = dataset3.batch(10)
for item in dataset3:
    print(item)

# apply
dataset4 = dataset3.apply(tf.data.experimental.unbatch())

# filter
dataset5 = dataset.filter(lambda x: x<7).batch(5)
for item in dataset5:
    print(item)
for item in dataset5.take(1):
    print(item)

# %%
# shuffle

dataset = tf.data.Dataset.range(10).repeat(-1) # 0 to 9, three times
dataset = dataset.shuffle(buffer_size=2, seed=42).batch(2)
for item in dataset.take(10):
    print(item)


#%%
# tfrecord

with tf.io.TFRecordWriter('my_data.tfrecord') as f:
    f.write(b'this is the first record')
    f.write(b'and this is the second record')

filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)


#%% 
# tensorflow protobufs

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

person_example = Example(
    features=Features(
        feature={
        "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
        "id": Feature(int64_list=Int64List(value=[123])),
        "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
}))

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())


feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}
for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)

print(parsed_example)


parsed_example["emails"].values[0]
tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
parsed_example["emails"].values

#%%
# putting images in TFRecords

from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

img = load_sample_images()["images"][0]
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()

data = tf.io.encode_jpeg(img)
example_with_image = Example(features=Features(feature={
    "image": Feature(bytes_list=BytesList(value=[data.numpy()]))}))
serialized_example = example_with_image.SerializeToString()

# then save to TFRecord
feature_description = { "image": tf.io.VarLenFeature(tf.string) }
example_with_image = tf.io.parse_single_example(serialized_example, feature_description)
decoded_img = tf.io.decode_image(example_with_image["image"].values[0])

plt.imshow(decoded_img)
plt.title("Decoded Image")
plt.axis("off")
plt.show()

# %%
# sequential data

FeatureList = tf.train.FeatureList
FeatureLists = tf.train.FeatureLists
SequenceExample = tf.train.SequenceExample

context = Features(feature={
    "author_id": Feature(int64_list=Int64List(value=[123])),
    "title": Feature(bytes_list=BytesList(value=[b"A", b"desert", b"place", b"."])),
    "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25]))
})

content = [["When", "shall", "we", "three", "meet", "again", "?"],
           ["In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"]]
comments = [["When", "the", "hurlyburly", "'s", "done", "."],
            ["When", "the", "battle", "'s", "lost", "and", "won", "."]]

def words_to_feature(words):
    return Feature(bytes_list=BytesList(value=[word.encode("utf-8")
                                               for word in words]))

content_features = [words_to_feature(sentence) for sentence in content]
comments_features = [words_to_feature(comment) for comment in comments]
            
sequence_example = SequenceExample(
    context=context,
    feature_lists=FeatureLists(feature_list={
        "content": FeatureList(feature=content_features),
        "comments": FeatureList(feature=comments_features)
    }))

print(sequence_example)



serialized_sequence_example = sequence_example.SerializeToString()

context_feature_descriptions = {
    "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "title": tf.io.VarLenFeature(tf.string),
    "pub_date": tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
}
sequence_feature_descriptions = {
    "content": tf.io.VarLenFeature(tf.string),
    "comments": tf.io.VarLenFeature(tf.string),
}
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example, context_feature_descriptions,
    sequence_feature_descriptions)

print(parsed_context)
print(parsed_context["title"].values)
print(parsed_feature_lists)

print(tf.RaggedTensor.from_sparse(parsed_feature_lists["content"]))

#%%
# standardization layer

import tensorflow as tf
from tensorflow import keras
import numpy as np

class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)

    def call(self, inputs):
        return (input - self.means_) / (self.stds_ + keras.backend.epsilon())


std_layer = Standardization()
std_layer.adapt(data_sample)

model = keras.Sequential()
model.add(std_layer)


#%%
# encoding categorical featuers using one-hot vectors
# try keras.layers.TextVectorization => will be same as below code

import tensorflow as tf
from tensorflow import keras
import numpy as np

vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)


categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)

cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab) + num_oov_buckets)
print(cat_one_hot)


#%%
# embedding

embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indices = table.lookup(categories)
print(cat_indices)

cat_emb = tf.nn.embedding_lookup(embedding_matrix, cat_indices)
print(cat_emb)

# try keras.layers.Embedding layer -> same as above
embedding_layer = keras.layers.Embedding(input_dim=len(vocab) + num_oov_buckets,
                                            output_dim=embedding_dim)

print(embedding_layer(cat_indices))


#%%
# multi input (1. normal input + embedding input)
regular_inputs = keras.layers.Input(shape=[8])

categoreis = keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)

encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
output = keras.layers.Dense(1)(encoded_inputs)

model = keras.models.Model(inputs=[regular_inputs, categoreis], output=[outputs])