import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow_addons.layers import CRF
from keras.utils import to_categorical
import pickle

#load data
df = pd.read_csv("ner.csv", encoding='latin1')
df = df.fillna(method="ffill")

#group sentences by sentence_id
class SentenceGetter:
    def __init__(self, data):
        self.data = data
        self.grouped = self.data.groupby("sentence_id").apply(
            lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        )
        self.sentences = [s for s in self.grouped]

getter = SentenceGetter(df)
sentences = getter.sentences

#build vocab
words = list(set(df["word"].values))
tags = list(set(df["tag"].values))

word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0

tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

#save mappings
np.save("wordindex.npy", word2idx)
np.save("tagindex.npy", tag2idx)
np.save("indextag.npy", idx2tag)

#encode sentences
MAX_LEN = 50
X = [[word2idx.get(w[0], 1) for w in s] for s in sentences]
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_LEN, padding="post")

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=MAX_LEN, padding="post")
y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#build model
model = Sequential()
model.add(Embedding(input_dim=len(word2idx), output_dim=64, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
crf = CRF(len(tag2idx))
model.add(crf)

model.compile(optimizer="adam", loss=crf.loss, metrics=[crf.accuracy])
model.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.1)

model.save("ner_model.h5")
print("Model trained and saved.")
