from sklearn import datasets, neighbors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class DS:
    def __init__(self):
        self.iris = datasets.load_iris()
        self.data = self.iris.data
        print(self.data.shape)

    def reshaping_example(self):
        """
        When the data is not in the format of (n_samples, n_features)
        it will need to be reshaped. An example of this is the digits
        dataset. This is made up of 1797 8x8 images of hand-written digits
        :return: reshaped dataset
        """
        digits = datasets.load_digits()

        print(digits.images.shape)

        plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)

        # To use this dataset with scikit, we transform each of the 8x8 image into
        # a vector of length 64

        data = digits.images.reshape((digits.images.shape[0], -1))
        return data

    def estimator_info(self):
        """
        The main API implemented by scikit-learn is that of the estimator. An estimator
        is an object that learns from data; it may be a classification, regression or a
        transformer that extracts/filters useful features from raw data.

        All estimators expose a fit method that takes a dataset (usually a 2-d array)

        estimator.fit(data)

        Estimator parameters: All the parametes of an estimator can be set when it is
        instantiated or when modifying the corresponding attribute.

        estimator = Estimator(param1=1, param2=2)
        estimator.param1

        Estimated parameters: when data is fitted with an estimator, parameters are
        estimated from the data at hand. All the estimated parameter are attributes
        of the estimator object ending in an underscore:

        estimator.estimated_param_
        """
        pass

class SupervisedLearning:
    """
    Supervised learning: predicting an output/response variable from high
    dimensional observations

    observed data: X in R^n
    target data: y {target, labels} in R1

    All supervised estimators in scikit-learn implement a fit(X,y) method and
    predict(X) method, given unlabled
    """

    def __init__(self):
        self.iris = datasets.load_iris()
        self.iris_X = self.iris.data
        self.iris_y = self.iris.target
        print("Unique classifiers: ", np.unique(self.iris_y))

    def nearest_neigbours_iris_example(self, n_neighbours):
        """
        Sample using nearest neighbours classification. It will plot the decision
        boundary for each class.
        :param n_neighbours:
        :return:
        """

        # import the first two features
        X = self.iris.data[:, :2]
        y = self.iris_y
        # Step size in the mesh
        h = 0.02

        # Create the colour maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        for weights in ['uniform', 'distance']:
            # create an instance of Neighbours Classifier and fit the data
            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbours, weights=weights)
            clf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a colour to each
            # point in the mesh [x_min, x_max]x[y_min, y_max]

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (n_neighbours, weights))
            plt.show()

    def split_train_test(self, data, target, estimator=neighbors.KNeighborsClassifier() ,rnd=np.random, set_seed=0):
        """
        This function takes a classifier and data and return a prediction and target for caomparison
        :param data, target, rnd, set_seed: data=dataset no target, target=target vector, 
                                           estimator: default KNeighboursClassifier() to match scikit learn example
                                           rnd(optional):default np.random, set_seed(optional): seed value
        :return y_train, y_test: return the train target and the test target for comparison
        """
        rnd.seed(set_seed)
        inidices = np.random.permutation(len(data))
        X_train = data[inidices[:-int(0.1*len(data))]]
        y_train = target[inidices[:-int(0.1*len(data))]]
        X_test = data[inidices[-int(0.1*len(data)):]]
        y_test = target[inidices[-int(0.1*len(data)):]]
        estimator.fit(X_train, y_train)
        
        # Create and fit a nearest neighbour estimator
        return estimator.predict(X_test)
        







