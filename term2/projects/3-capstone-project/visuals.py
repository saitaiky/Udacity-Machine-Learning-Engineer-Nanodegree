###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.learning_curve as curves
#from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# too Slow...
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import accuracy_score, confusion_matrix
#from sklearn.utils.multiclass import unique_labels

from sklearn.cross_validation import ShuffleSplit, train_test_split
#from sklearn.model_selection import train_test_split, ShuffleSplit




#Common Model Algorithms
from sklearn import svm, tree, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics



def DecisionTree_ModelLearning(X, y, rawImages, model_name):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    #train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    # Note that for classification the number of samples usually have to be big enough to contain at least one sample from each class.
    train_sizes = np.rint(np.linspace(X.shape[0]*0.01, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, depth in enumerate([2, 4, 5, 8]):

        # Create a Decision tree classifier at max_depth = depth
        classifier = DecisionTreeClassifier(max_depth = depth)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(classifier, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        print(test_mean)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Classifier Learning Performances', fontsize = 16, y = 1.03)
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.5)
    fig.show()


    trainFeat, testFeat, trainLabels, testLabels = train_test_split(X,y,test_size=0.25,random_state=42)
    trainRI, testRI, trainRL, testRL = train_test_split(rawImages, y, test_size=0.25, random_state=42)
    trainFeat = trainFeat.astype('int')
    testFeat = testFeat.astype('int')

    start_time = time.time()
    neigh = DecisionTreeClassifier(max_depth = 5)
    neigh.fit(trainFeat, trainLabels)
    testt = neigh.predict(testFeat)
    score = accuracy_score(testLabels, testt)
    print("Decision Tree (max_depth=5) feature testing accuracy=", score*100)
    neigh.fit(trainRI, trainRL)
    test_pred = neigh.predict(testRI)
    score = accuracy_score(testRL, test_pred)
    print("Decision Tree (max_depth=5) raw testing accuracy=", score*100)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_confusion_matrix(testLabels, test_pred, normalize=True, title=model_name)


def DecisionTree_ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    max_depth = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(DecisionTreeClassifier(), X, y, \
        param_name = "max_depth", param_range = max_depth, cv = cv, scoring = 'accuracy')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Classifier Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(max_depth, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(max_depth, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()

def KNeighbors_ModelLearning(X, y, rawImages, model_name):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    #train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    # Note that for classification the number of samples usually have to be big enough to contain at least one sample from each class.
    train_sizes = np.rint(np.linspace(X.shape[0]*0.01, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, n_neighbors in enumerate([1, 3, 6, 9]):

        # Create a Decision tree classifier at max_depth = depth
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto')

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(neigh, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        print(test_mean)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('n_neighbors = %s'%(n_neighbors))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('KNeighbors Classifier Learning Performances', fontsize = 16, y = 1.03)
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.5)
    fig.show()


    trainFeat, testFeat, trainLabels, testLabels = train_test_split(X,y,test_size=0.25,random_state=42)
    trainRI, testRI, trainRL, testRL = train_test_split(rawImages, y, test_size=0.25, random_state=42)
    trainFeat = trainFeat.astype('int')
    testFeat = testFeat.astype('int')

    start_time = time.time()
    neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    neigh.fit(trainFeat, trainLabels)
    test_pred = neigh.predict(testFeat)
    score = accuracy_score(testLabels, test_pred)
    print("KNN (n_neighbors=5) feature testing accuracy=", score*100)
    #neigh.fit(trainRI, trainRL)
    #testt = neigh.predict(testRI)
    #score = accuracy_score(testRL, testt)
    #print("KNN (n_neighbors=5) raw testing accuracy=", score*100)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_confusion_matrix(testLabels, test_pred, normalize=True, title=model_name)


def KNeighbors_ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    n_neighbors = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(KNeighborsClassifier(weights='uniform', algorithm='auto'), X, y, \
        param_name = "n_neighbors", param_range = n_neighbors, cv = cv, scoring = 'accuracy')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('KNeighbors Complexity Performance')
    plt.plot(n_neighbors, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(n_neighbors, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(n_neighbors, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(n_neighbors, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()


def SVM_ModelLearning(X, y, rawImages, model_name):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    #train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    # Note that for classification the number of samples usually have to be big enough to contain at least one sample from each class.
    train_sizes = np.rint(np.linspace(X.shape[0]*0.01, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, gamma in enumerate([1, 10, 100,1000]):

        # Create a Decision tree classifier at max_depth = depth
        svc = SVC(kernel="rbf", C=.53, gamma=gamma)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(svc, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        print(test_mean)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('gamma = %s'%(gamma))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('SVC Learning Performances', fontsize = 16, y = 1.03)
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.5)
    fig.show()


    trainFeat, testFeat, trainLabels, testLabels = train_test_split(X,y,test_size=0.25,random_state=42)
    trainRI, testRI, trainRL, testRL = train_test_split(rawImages, y, test_size=0.25, random_state=42)
    trainFeat = trainFeat.astype('int')
    testFeat = testFeat.astype('int')

    start_time = time.time()
    svc = SVC(kernel="rbf", C=.53, gamma=10)
    svc.fit(trainFeat, trainLabels)
    test_pred = svc.predict(testFeat)
    score = accuracy_score(testLabels, test_pred)
    print("SVC (C=.53, gamma=10) feature testing accuracy=", score*100)
    #svc.fit(trainRI, trainRL)
    #testt = svc.predict(testRI)
    #score = accuracy_score(testRL, testt)
    #print("SVC (C=.53, gamma=10) raw testing accuracy=", score*100)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_confusion_matrix(testLabels, test_pred, normalize=True, title=model_name)


def SVM_ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    gamma = np.linspace(1, 10, 10)
    # c_value = np.linspace(.5, .53, 10)
    #gamma = [10, 100, 1000, 10000, 100000]

    # Calculate the training and testing scores
    train_scores, test_scores = curves.validation_curve(SVC(kernel="rbf", C=.53), X, y, \
        param_name = "gamma", param_range = gamma, cv = cv, scoring = 'accuracy')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('SVM Complexity Performance')
    plt.plot(gamma, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(gamma, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(gamma, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(gamma, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Gamma')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()





def AdaBoost_ModelLearning(X, y, rawImages, model_name):
    """ Calculates the performance of several models with varying sizes of training data.
        The learning and testing scores for each model are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    #train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)
    # Note that for classification the number of samples usually have to be big enough to contain at least one sample from each class.
    train_sizes = np.rint(np.linspace(X.shape[0]*0.01, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10,7))

    # Create three different models based on max_depth
    for k, n_estimators in enumerate([10, 20, 30, 40]):

        # Create a Decision tree classifier at max_depth = depth
        estimatorCart = DecisionTreeClassifier(max_depth=1)
        abc = AdaBoostClassifier(base_estimator=estimatorCart, n_estimators=n_estimators, learning_rate=.001)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = curves.learning_curve(abc, X, y, \
            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        print(test_mean)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('n_estimators = %s'%(n_estimators))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('AdaBoost Classifier Learning Performances', fontsize = 16, y = 1.03)
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.5)
    fig.show()


    trainFeat, testFeat, trainLabels, testLabels = train_test_split(X,y,test_size=0.25,random_state=42)
    #trainRI, testRI, trainRL, testRL = train_test_split(rawImages, y, test_size=0.25, random_state=42)
    trainFeat = trainFeat.astype('int')
    testFeat = testFeat.astype('int')

    start_time = time.time()
    estimatorCart = DecisionTreeClassifier(max_depth=2)
    abc = AdaBoostClassifier(base_estimator=estimatorCart, n_estimators=40, learning_rate=.05)
    abc.fit(trainFeat, trainLabels)
    test_pred = abc.predict(testFeat)
    score = accuracy_score(testLabels, test_pred)
    print("AdaBoost (n_estimators=40) feature testing accuracy=", score*100)
    #abc.fit(trainRI, trainRL)
    #testt = abc.predict(testRI)
    #score = accuracy_score(testRL, testt)
    #print("AdaBoost (max_iter_predict=100) raw testing accuracy=", score*100)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_confusion_matrix(testLabels, test_pred, normalize=True, title=model_name)


def AdaBoost_ModelComplexity(X, y):
    """ Calculates the performance of the model as model complexity increases.
        The learning and testing errors rates are then plotted. """

    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

    n_estimators = np.rint(np.linspace(1, 50, 10)).astype(int)

    # Calculate the training and testing scores
    estimatorCart = DecisionTreeClassifier(max_depth=1)
    abc = AdaBoostClassifier(base_estimator=estimatorCart, learning_rate=.05)
    train_scores, test_scores = curves.validation_curve(abc, X, y, \
        param_name = "n_estimators", param_range = n_estimators, cv = cv, scoring = 'accuracy')

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('AdaBoost Complexity Performance')
    plt.plot(n_estimators, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(n_estimators, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(n_estimators, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(n_estimators, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()


def gridSearch_tunning(X, y, rawImages):

    # Create 10 cross-validation sets for training and testing
    cv_split = ShuffleSplit(X.shape[0], n_iter = 10, test_size = .3,  train_size = .6,random_state = 0)

    vote_est = [
        #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc',ensemble.ExtraTreesClassifier()),
        ('gbc', ensemble.GradientBoostingClassifier()),
        ('rfc', ensemble.RandomForestClassifier()),

        #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
        ('gpc', gaussian_process.GaussianProcessClassifier()),

        #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
        ('bnb', naive_bayes.BernoulliNB()),
        ('gnb', naive_bayes.GaussianNB()),

        #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
        ('knn', neighbors.KNeighborsClassifier()),

        #SVM: http://scikit-learn.org/stable/modules/svm.html
        ('svc', svm.SVC(probability=True)),
    ]

    #Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]


    grid_param = [
                [{
                #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
                'n_estimators': grid_n_estimator, #default=50
                'learning_rate': grid_learn, #default=1
                #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
                'random_state': grid_seed
                }],


                [{
                #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
                'n_estimators': grid_n_estimator, #default=10
                'max_samples': grid_ratio, #default=1.0
                'random_state': grid_seed
                 }],


                [{
                #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
                'n_estimators': grid_n_estimator, #default=10
                'criterion': grid_criterion, #default=”gini”
                'max_depth': grid_max_depth, #default=None
                'random_state': grid_seed
                 }],


                [{
                #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
                #'loss': ['deviance', 'exponential'], #default=’deviance’
                'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
                #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
                'max_depth': grid_max_depth, #default=3
                'random_state': grid_seed
                 }],


                [{
                #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
                'n_estimators': grid_n_estimator, #default=10
                'criterion': grid_criterion, #default=”gini”
                'max_depth': grid_max_depth, #default=None
                'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
                'random_state': grid_seed
                 }],

                [{
                #GaussianProcessClassifier
                'max_iter_predict': grid_n_estimator, #default: 100
                'random_state': grid_seed
                }],


                [{
                #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
                'alpha': grid_ratio, #default: 1.0
                 }],


                #GaussianNB -
                [{}],

                [{
                #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
                'n_neighbors': [1,2,3,4,5,6,7], #default: 5
                'weights': ['uniform', 'distance'], #default = ‘uniform’
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }],


                [{
                #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
                #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1,2,3,4,5], #default=1.0
                'gamma': grid_ratio, #edfault: auto
                'decision_function_shape': ['ovo', 'ovr'], #default:ovr
                'probability': [True],
                'random_state': grid_seed
                 }],

            ]

    #trainFeat, testFeat, trainLabels, testLabels = train_test_split(X,y,test_size=0.25,random_state=42)
    #trainRI, testRI, trainRL, testRL = train_test_split(rawImages, y, test_size=0.25, random_state=42)
    trainFeat = X.astype('int')
    #testFeat = testFeat.astype('int')

    start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
    for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

        #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
        #print(param)
        start = time.perf_counter()
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'accuracy')
        best_search.fit(trainFeat, y)
        run = time.perf_counter() - start

        best_param = best_search.best_params_
        print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
        clf[1].set_params(**best_param)


    run_total = time.perf_counter() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

    print('-'*10)



def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)

        # Fit the data
        reg = fitter(X_train, y_train)

        # Make a prediction
        pred = reg.predict([data[0]])[0]
        prices.append(pred)

        # Result
        print("Trial {}: ${:,.2f}".format(k+1, pred))

    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    classes = ['crazy', 'inclusion', 'patches', 'pitted surface', 'rolled-in scale', 'scratches']

    title = title + " normalized confusion matrix"
    #if not title:
    #    if normalize:
    #        title = 'Normalized confusion matrix'
    #    else:
    #        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(1, 1, 1)
    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
