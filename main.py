from tensorflow import keras
import sklearn
from sklearn.preprocessing import *
import data_reader
import svm
import DMLP
import MLPC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Global Variable
EPOCHS = 200

# data reader Initialization
dr = data_reader.DataReader()

print("데이터 셋 형태 : \n feature : {0} \n label : {1} \n instance : {2}"
      .format(dr.train_X.shape[1],
              len(set(dr.train_Y.ravel())),
              dr.train_X.shape[0] + dr.test_X.shape[0]))

# dataset Normalization
dr.train_X = StandardScaler().fit_transform(dr.train_X)
dr.test_X = StandardScaler().fit_transform(dr.test_X)

# model Initialization
svm = svm.SVC().model
mlp = MLPC.MLPC().model
dmlp = DMLP.DMLP().model

# Callback function
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# model Training
mlp_his = mlp.fit(dr.train_X, dr.train_Y.ravel(), epochs=EPOCHS, batch_size=512,
                  validation_split=0.1,
                  callbacks=[early_stop])

dmlp_his = dmlp.fit(dr.train_X, dr.train_Y.ravel(), epochs=EPOCHS, batch_size=512,
                    validation_split=0.1,
                    callbacks=[early_stop])

svm_model = svm.fit(dr.train_X, dr.train_Y.ravel())

# Draw time graph - Loss, Accuracy
history_df = pd.DataFrame(mlp_his.history)
history_df['accuracy'].plot()
history_df['val_accuracy'].plot()
plt.title('MLP Accuracy')
plt.ylabel('accuracy')
plt.ylim([0.5, 1.0])
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

history_df = pd.DataFrame(dmlp_his.history)
history_df['accuracy'].plot()
history_df['val_accuracy'].plot()
plt.title('Deep MLP Accuracy')
plt.ylabel('accuracy')
plt.ylim([0.5, 1.0])
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# SVM Predict
svm_pre = svm_model.predict(dr.test_X)

svm_score = sklearn.metrics.accuracy_score(svm_pre, dr.test_Y.ravel())
svm_report = sklearn.metrics.classification_report(svm_pre, dr.test_Y.ravel(), target_names=['normal', 'spam'])

# MLP Predict
mlp_pre = mlp.predict(dr.test_X)
mlp_pre = np.where(mlp_pre > 0.5, 1, 0)

mlp_score = sklearn.metrics.accuracy_score(mlp_pre, dr.test_Y.ravel())
mlp_report = sklearn.metrics.classification_report(mlp_pre, dr.test_Y.ravel(), target_names=['normal', 'spam'])

# Deep MLP Predict
dmlp_pre = dmlp.predict(dr.test_X)
dmlp_pre = np.where(dmlp_pre > 0.5, 1, 0)

dmlp_score = sklearn.metrics.accuracy_score(dmlp_pre, dr.test_Y.ravel())
dmlp_report = sklearn.metrics.classification_report(dmlp_pre, dr.test_Y.ravel(), target_names=['normal', 'spam'])

# Score
print(f"""
SVM Accuracy : {svm_score}
{svm_report}

MLP Accuracy : {mlp_score}
{mlp_report}

Deep MLP Accuracy : {dmlp_score}
{dmlp_report}
""")
