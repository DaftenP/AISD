from tensorflow import keras
import data_reader


EPOCHS = 200

dr = data_reader.DataReader()


print(dr.train_X.shape)