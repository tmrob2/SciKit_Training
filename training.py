"""A section on basic training with scikit so I remember what I am doing"""

from sklearn import random_projection
from sklearn import datasets
import numpy as np
from sklearn import svm
import pickle
from sklearn.externals import joblib
from sklearn.svm import SVC

class basic_training:
    """A section on conventions"""

    def __init__(self, Print=False):
        """Import the relevant data sets
           A dataset is a dictionary like object which holds: data, metadata
           datasets are store in .data
           n_samples, n_features array
           in the case of a supervised problem one or more response variables are stored in .target

           e.g. to get the data from digits interpret: self.digits.data

           e.g. digits.target gives the 'ground truth' for the digits dataset
           :arg: Print(type - Boolean) allows the printing of examples such that we can see how the data
           gets manipulated
        """
        self.iris = datasets.load_iris()
        self.digits = datasets.load_digits()
        if Print:
            # Print examples
            print('Dataset data: \n', self.digits.data)
            print('Target data: \n', self.digits.target)

            # Print accessing data example with scikit
            print('Accessing data is via digits.image[i] where i in n_features \n', self.digits.images[0])

            # Alternative data slicing procedure
            print('An alternative process for accessing data is via digits.data[:][i] which achieves the same result \n',
                  self.digits.data[:][1])

    def learning_predicting(self):
        """Given the task is to predict given a row which target variable does it represent
           E.g. SVM:
           gamma: in this example we set gamma manually. It is possible to automatically find good values
           for the parameters by using tools such as grid search and cross validation
           :return: pickle model dump of the classifier, clf which is the classifier
        """
        clf = svm.SVC(gamma=0.001, C=100.)

        # Pass the training set to the fit method, let the training set be all of the images except
        # for the last one i.e. [:-1]
        s = pickle.dumps(clf)

        return s, clf

    def model_persitence(self, option='pickle', filename=None):
        """
        It is possible to save a model in scikit by using Python's built in persistence model, namely pickle
        :arg: option takes a string of 'pickle' or a string 'joblib', this defines which type is used for model
        storage i.e. pickle for in memory storage or joblib for big data;
        :return: a pickle dump of the prediction model, and the classification;
        """
        clf = svm.SVC()
        X, y = self.iris.data, self.iris.target
        clf.fit(X, y)
        if option == 'pickle':
            s = pickle.dumps(clf)
            return s, clf
        else:
            joblib.dump(clf, filename+'.pkl')
            return None, clf

    def pickle_load_big(self, filename):
        """Loading a presaved pickle from file"""
        clf = joblib.load(filename+'.pkl')
        return clf

class conventions_example:
    rng = np.random.RandomState(0)
    # set some random data
    X = rng.rand(10, 2000)
    # Type cast the array to a specific type in this example float32
    X = np.array(X, dtype='float32')
    transformer = random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X)

    def __init__(self):
        print(self.X.dtype)
        print(self.X_new.dtype)
        print('X is float32 which has been cast to float64 by fit_transform(X)')

        print('Regression targets are cast to float64, classification targets are maintained')
        self.iris = datasets.load_iris()
        self.clf = SVC()
        self.clf.fit(self.iris.data, self.iris.target)

    def refitting_updating(self):
        self.X = self.rng.rand(100, 10)
        self.y = self.rng.binomial(1, 0.5, 100)
        self.X_test = self.rng.rand(5, 10)
        clf = SVC()

        # Example classifier linear
        clf.set_params(kernal='linear').fit(self.X, self.y)
        print(clf)
        print(clf.predict(self.X_test))

        # Example classifier rbf
        clf.set_params(kernal='rbf').fit(self.X, self.y)
        print(clf)
        print(clf.predict(self.X_test))














