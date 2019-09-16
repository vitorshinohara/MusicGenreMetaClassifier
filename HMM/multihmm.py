import copy
import numpy as np

class MultipleHMM():
    base_model = None
    models = {}

    def __init__(self, base_model=None):
        self.base_model = base_model

    def fit(self, X, y, train_lengths):
        """Fits all internal models"""
        labels = set(y)
        for l in labels:
            l_index = [i for i in range(len(y)) if y[i] == l]
            if X[0].ndim == 1:
                this_x = [X[i].reshape(-1, 1) for i in l_index]
            else:
                this_x = [X[i] for i in l_index]

            this_lengths = [train_lengths[i] for i in l_index]

            my_x = this_x[0]
            for i in range(1, len(this_x)):
                my_x = np.vstack ((my_x, np.array(this_x[i])))

            new_model = copy.deepcopy(self.base_model)
            new_model.fit(my_x, this_lengths)
            self.models[l] = new_model

    def predict(self, X):
        """Predicts a label for input X"""
        return_label = None
        best_prob = None
        for label in self.models.keys():

            #print "Computing probabilities on shape", X.shape
            if X.ndim == 1:
                this_prob = self.models[label].score(X.reshape(-1,1))
            else:
                this_prob = self.models[label].score(X)

            #print "Prob in label", label, "=", this_prob
            if (best_prob is None) or (this_prob > best_prob):
                best_prob = this_prob
                return_label = label
        return return_label


