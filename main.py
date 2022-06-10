from tensorflow import keras
import data_reader
from sklearn import svm

EPOCHS = 200

dr = data_reader.DataReader()

print(dr.train_X.shape)

s = svm.SVC().fit(dr.train_X, dr.train_Y)

res = s.predict(dr.test_X)

print(res.score(dr.test_X, dr.test_Y))