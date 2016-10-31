import numpy as np
from sklearn import datasets

class data:
    """ This class is designed to set up the data on initialisation.
        We will be working with the iris data set and following the examples
        on the scikit-learn website

        insert citation
    """
    def __init__(self):
        self.iris = datasets.load_iris()
        self.iris_X = self.iris.data
        self.iris_y = self.iris.target

    def create_training_and_testing(self):

