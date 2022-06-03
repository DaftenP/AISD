from sklearn import svm


# dr = data_reader.DataReader()
#
# s = svm.SVC(gamma=0.001)
# s.fit(dr.train_X, np.ravel(dr.train_Y))
#
# res = s.predict(dr.test_X)
#
#
# print("정확도 : ", s.score(dr.test_X, dr.test_Y), "%")


class SVM():
    def __init__(self):
        self.model = self.new_model()

    def new_model(self):
        model = svm.SVC(gamma=0.001)

        return model
