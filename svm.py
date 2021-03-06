from sklearn import svm

class SVC():
    def __init__(self):
        self.model = self.new_model()

    def new_model(self):
        model = svm.SVC(gamma=0.001,
                        kernel='linear',
                        C=10,
                        verbose=1)

        return model
