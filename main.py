from tensorflow import keras
import sklearn
from sklearn.preprocessing import *
import data_reader
import svm
import DMLP
import MLPC
import matplotlib.pyplot as plt
import pandas as pd

EPOCHS = 200

dr = data_reader.DataReader()

print("데이터 셋 형태 : \n feature : {0} \n label : {1} \n instence : {2}"
      .format(dr.train_X.shape[1],
              len(set(dr.train_Y.ravel())),
              dr.train_X.shape[0] + dr.test_X.shape[0]))


dr.train_X = StandardScaler().fit_transform(dr.train_X)
dr.test_X = StandardScaler().fit_transform(dr.test_X)

svm = svm.SVC().model
mlp = MLPC.MLPC().model
dmlp = DMLP.DMLP().model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

mlp_his = mlp.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, batch_size=512,
                  validation_data=(dr.test_X, dr.test_Y),
                  callbacks=[early_stop])

dmlp_his = dmlp.fit(dr.train_X, dr.train_Y, epochs=EPOCHS, batch_size=512,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

svm_model = svm.fit(dr.train_X, dr.train_Y.ravel())
pre = svm_model.predict(dr.test_X)

score = sklearn.metrics.accuracy_score(pre, dr.test_Y.ravel())
report = sklearn.metrics.classification_report(pre, dr.test_Y.ravel(), target_names=['normal', 'spam'])

print("Accuracy : {0}" .format(score))
print(report)

history_df = pd.DataFrame(mlp_his.history)
history_df['accuracy'].plot()
history_df['val_accuracy'].plot()
plt.title('MLP Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

history_df = pd.DataFrame(dmlp_his.history)
history_df['accuracy'].plot()
history_df['val_accuracy'].plot()
plt.title('Deep MLP Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
