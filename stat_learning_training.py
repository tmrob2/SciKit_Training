from __future__ import print_function
from sklearn import datasets, neighbors, linear_model, svm
from sklearn.svm import SVC
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

    Some key points:
    Scikit provides Lasso object which solves tha lasso regression problem
    using coordinate descent, this is most efficient on very large datasets

    Scikit also provides the LassoLars object which is efficient for problems
    in which the weight vector estimates is very sparse
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
        
        # Create and fit a nearest neighbour estimator
        return X_train, y_train, X_test, y_test, estimator

    def diabetes_example(self):
        """
        Load the diabetes dataset
        :return: diabetes_X_train, diabetes_X_test,
        diabetes_y_train, diabetes_y_test
        """
        diabetes = datasets.load_diabetes()
        diabetes_X_train = diabetes.data[:-20]
        diabetes_X_test = diabetes.data[-20:]
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        return diabetes_X_train, diabetes_X_test, \
               diabetes_y_train, diabetes_y_test

    def fit_linear_model(self, data_train, target_train, data_test,
                         target_test, model=linear_model.LinearRegression()):
        """
        Fits a linear model, however takes a model as a parameter so could
        effectively change this to whatever, but would probably result in
        an error because it may or may not have the appropriate attributes
        :param data_train: training data R^2
        :param target_train: target data R
        :param data_test: testing data R^2
        :param target_test: target evaluation against prediction R
        :param model: a linear classifier defaults at linear model
        :return: returns the model for further evaluation of attributes
        """
        model.fit(data_train, target_train)
        print(model.coef_)
        print(model.score(data_test, target_test))
        return model

    def shrinkage_example(self):
        """
        If there are few data points per dimension, noise in the observation
        induces high variance. A solution to high-dimension statistical learning
        is to shrink the regression coefficients to zero: any two randomly chosen
        set of observations are likely to be uncorrelated. This is calles ridge
        regression.
        :return:
        """

        # c_ is equivalent to cbind
        X = np.c_[0.5, 1].T
        y = [0.5, 1]
        test = np.c_[0, 2].T
        regr = linear_model.LinearRegression()

        plt.figure()

        np.random.seed(0)
        for _ in range(6):
            this_X = 0.1*np.random.normal(size=(2, 1)) + X
            regr.fit(this_X, y)
            plt.plot(test, regr.predict(test))
            plt.scatter(this_X, y, s=3)

        # This is an example of the bias/variance trade off
        # the larger the value of alpha the higher the bias
        # and the lower the variance

        regr_r = linear_model.Ridge(alpha=0.1)
        plt.figure()
        np.random.seed(0)
        for _ in range(6):
            this_X = 0.1*np.random.normal(size=(2, 1)) + X
            regr_r.fit(this_X, y)
            plt.plot(test, regr_r.predict(test))
            plt.scatter(this_X, y, s=3)

    def ridge_regression_example(self, train_data, test_data, train_target,
                                 test_target, model=linear_model.Ridge()):
        alphas = np.logspace(-4, -1, 6)

        print([model.set_params(alpha=alpha
                                 ).fit(train_data, train_target
                                       ).score(test_data, test_target)
               for alpha in alphas])

    def sparse_methods_example(self, train_data, test_data,
                               train_target, test_target,
                               model=linear_model.Lasso()):
        """
        This approach is called Lasso: least absolute shrinkage and selction
        operator. This helps deal with the dimension problem. i.e. running
        through a large amound of permutations to get an answer
        :return:
        """
        alphas = np.logspace(-4, -1, 6)
        scores = [model.set_params(alpha=alpha).fit(train_data, train_target
                                                    ).score(test_data, test_target)
                  for alpha in alphas]
        best_alpha = alphas[scores.index(max(scores))]

        print(best_alpha)
        model.alpha = best_alpha
        model.fit(test_data, test_target)
        print(model.coef_)
        return model

    def classification_example(self, X, X_test, y, y_test,
                               model1, model2):
        """
        here C in the default model is the inverse of the regularisation strength
        :param X:
        :param X_train:
        :param y:
        :param y_train:
        :param model:
        :return:
        """
        print("Model 1 predictive score: %f", model1.fit(X, y).score(X_test, y_test))
        print("Model 2 predictive score: %f", model2.fit(X, y).score(X_test, y_test))

        return model1, model2

    def plotting_decision_surface(self, in_dataset):
        """
        Plotting the decision surface for classifiers
        The linear models LinearSVC() and SVC(kernal='linear') yield
        slightly different decision boundaries. This can be a consequence
        of the following differences:
        LinearSVC: minimises the squared hinge loss while SVC minimises
        the regular hinge loss.
        LinearSVC uses the One-vs-All multiclass reduction while SVC uses
        the One-vs-One multiclass reduction

        Both linear models have linear decision boundaries (intersecting
        hyperplanes) while the non-linear kernal models (polynomial or
        Gaussian RBF) have more flexible non-linear decision boundaries
        with shapes that depend on the kind of kernal and its parameters

        :return:
        """

        X = in_dataset.data[:, :2] #take the first two features of the dataset
        y = in_dataset.target
        h = 0.02 #Step size

        # Ceate an instance of SVM and fit our data.
        C = 1.0 # SVM regularisation parameter
        # rbf - radial basic function
        svc = SVC(kernel='linear', C=C).fit(X, y)
        rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
        poly_svc = SVC(kernel='poly', degree=3).fit(X, y)
        lin_svc = svm.LinearSVC(C=C).fit(X, y)

        # create a Mesh plot
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        titles = ['SVC with linear kernal',
                  'LinearSVC (linear kernal)',
                  'SVC with RBF kernal',
                  'SVC with polynomial (degree 3) kernal']

        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            # Plot the decision boundary. Assign a colour for each
            # point in the mesh [x_min, x_max].[y_min, y_max]
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # put the result into a colour plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

            # Plot the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])

        plt.show()
class Support_Vector_Machines:
    """
    Support vector machines belong to a discriminant model family: they try
    to find a combination of samples to build a plane maximising the margin
    between the two classes.

    Regularisation is set by the parameter c: a small value for c means the
    margin is claculated using many or all of the observations around the
    seperating line (more regularisation); a large value for c means the
    margin is calculated on observations close to the seperating line (less
    regularisation)
    """

    def regression_support_vector_machines(self):
        pass

    def classification_support_vector_machines(self):








