import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow_addons.layers import CRF
from utils import encode_sequences

#load model and mappings
model = load_model("ner_model.h5", custom_objects={"CRF": CRF})
wordindex = np.load("wordindex.npy", allow_pickle=True).item()
tagindex = np.load("tagindex.npy", allow_pickle=True).item()
indextag = np.load("indextag.npy", allow_pickle=True).item()

#load and preprocess test data again
import pandas as pd
df = pd.read_csv("ner.csv", encoding='latin1').fillna(method="ffill")

class SentenceGetter:
    def __init__(self, data):
        self.grouped = data.groupby("sentence_id").apply(
            lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        )
        self.sentences = [s for s in self.grouped]

getter = SentenceGetter(df)
sentences = getter.sentences

MAX_LEN = 50
X_test = [[wordindex.get(w[0], 1) for w in s] for s in sentences]
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_LEN, padding="post")
y_true = [[tagindex[w[1]] for w in s] for s in sentences]
y_true = tf.keras.preprocessing.sequence.pad_sequences(y_true, maxlen=MAX_LEN, padding="post")

#predict
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

#flatten and remove padding
y_true_flat, y_pred_flat = [], []
for i in range(len(y_true)):
    for j in range(MAX_LEN):
        if X_test[i][j] != 0:
            y_true_flat.append(indextag[y_true[i][j]])
            y_pred_flat.append(indextag[y_pred[i][j]])

print(classification_report(y_true_flat, y_pred_flat))
