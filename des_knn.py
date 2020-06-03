import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class DES_KNN(object):
    """This method selects an ensemble of classifiers. First of all it uses bagging to fit the data to base classifiers. 
    Then K-nearest neighbors are selected for the point. All classifiers are next fed with the neighbors in order to 
    calculate diversity and accuracy. N most accurate classifiers are selected. From them J most diverse are selected
    to create final ensemble that can predict the sample based on majority voting

    Arguments:
        base_estimator {BaseEstimator} -- base estimator that will be later used for prediction
        n_estimators {int} -- number of base estimators
        k {int} -- number of neighbors to estimate competence region
        N {int/float} -- int: number of N most accurate classifiers float: percentage of most accurate classifiers from n_estimators
        J {int/float} -- int: number of J most diverse classifiers float: percentage of most accurate classifiers from n_estimators
        max_samples {int/float} -- maximum samples amount to choose for bagging
        div_method {string} -- method to calculate diversity between pair of classifiers. Supported: Q, double-fault, q(correalation coefficient), disagreement
        knn_metrics {string} -- metrics to calculate distance between samples. Supported: euclidean, manhattan
        random_state {int} -- random_state to pass to base_estimator
    """

    def __init__(
        self,
        base_estimator=DecisionTreeClassifier(),
        n_estimators=10,
        k=7,
        N=0.5,
        J=0.3,
        max_samples=1.0,
        div_method="double-fault",
        knn_metrics="euclidean",
        random_state=77,
    ):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.k = k  # number of neighbors
        self.J = J  # J most diverse classifiers
        self.N = N  # N most accurate classifiers (J < N)
        self.max_samples = max_samples  # bagging
        self.div_method = div_method
        self.knn_metrics = knn_metrics
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, Y):
        # copy the dataset
        self.X = X
        self.Y = Y
        if len(X) != len(Y):
            raise ValueError("Wrong shape")
        self.estimators_ = []
        # bagging
        for i in range(self.n_estimators):
            # datasets to feed estimators
            X_ = []
            Y_ = []
            # if max_samples is int then choose max_samples samples
            if type(self.max_samples) is int:
                samples = np.random.randint(0, len(X) -1, size=self.max_samples)
                X_ = (X[samples])
                Y_ = (Y[samples])
            # if max_samples is float choose percentage of samples
            elif type(self.max_samples) is float:
                samples = np.random.randint(0, len(X) -1, size=int(self.max_samples * len(X)))
                X_ = (X[samples])
                Y_ = (Y[samples])
            # create estimator pool
            self.estimators_.append(clone(self.base_estimator).fit(X_, Y_))

    def find_k_nn(self, point, neighbors=None, metrics=None):
        """Function finds "neighbors" nearest neighbors based on the given point and metrics

        Arguments:
            point {array} -- neighbors are found based on it 

        Keyword Arguments:
            neighbors {int} -- the amount of closest neighbors to find (default: self.k)
            type {string} -- metrics used to determine distance (default: self.knn_metrics)

        Returns:
            list, list -- return lists of samples and coresponding classes
        """
        # get default args
        if metrics is None:
            metrics = self.knn_metrics
        if neighbors is None:
            neighbors = self.k
        # all distances
        distances = []
        # enumerate through all samples
        for i, row in enumerate(self.X):
            # calculate based on choosen metrics
            if metrics == "euclidean":
                dist = np.linalg.norm(np.array(point) - row)
            elif metrics == "manhattan":
                dist = np.sum(np.abs(np.array(point) - row))
            # append to distances
            distances.append([i, dist])
        # sort them based on distance
        distances.sort(key=lambda x: x[1])
        # choose k nearest ones
        neighbors = distances[:neighbors]
        # returns indices of k nearest neighbors
        x = [self.X[item[0]] for item in neighbors]
        y = [self.Y[item[0]] for item in neighbors]

        return x, y

    def calculate_accuracy(self, classifier, Xsamples, Ysamples):
        """Calculates the accuracy of given classifier

        Arguments:
            classifier {BaseEstimator} -- classifier which accuracy will be calculated
            Xsamples {np.array} -- list of samples
            Ysamples {np.array} -- list of corresponding classes

        Returns:
            float -- returns accuracy of classifier 
        """
        TP = 0  # True Positive
        TN = 0  # True Negative
        FP = 0  # False Positive
        FN = 0  # False Negative       
        # predict samples
        pred = classifier.predict(Xsamples)
        # choose right case
        for i, sample in enumerate(pred):    
            if sample == Ysamples[i] == 1:
                TP += 1
            elif sample == 0 and Ysamples[i] == 1:
                FN += 1
            elif sample == 1 and Ysamples[i] == 0:
                FP += 1
            elif sample == Ysamples[i] == 0:
                TN += 1
        # calculate accuracy
        accuracy = (TP + FN) / (TP + TN + FP + FN)

        return accuracy

    def calculate_pair_diversity(self, class1, class2, X, Y, method="double-fault"):
        """Calculates diversity between two classifiers based on selected method

        Arguments:
            class1 {BaseEstimator} -- 1st estimator
            class2 {BaseEstimator} -- 2nd estimator
            X {np.array} -- samples based on which diversity will be calculated
            Y {np.array} -- corresponding classes

        Keyword Arguments:
            method {str} -- method of calculating diversity (default: {"double-fault"})

        Returns:
            float -- return the diversity between two classifiers
        """
        N11 = 0  # both right
        N10 = 0  # 1 right, 2 wrong
        N01 = 0  # 1 wrong, 2 right
        N00 = 0  # both wrong
        
        # predict by both classifiers
        pred1 = class1.predict(X)
        pred2 = class2.predict(X)
        # check if predictions were correct
        # calculate Nxx
        for i in range(len(X)):    
            
            if pred1[i] == pred2[i] == Y[i]:
                N11 += 1
            elif pred1[i] == Y[i] and pred2[i] != Y[i]:
                N10 += 1
            elif pred1[i] != Y[i] and pred2[i] == Y[i]:
                N01 += 1
            elif pred1[i] == pred2[i] != Y[i]:
                N00 += 1
        # calculate based on given method
        if method == "Q":
            diversity = ((N11 * N00) - (N01 * N10)) / ((N11 * N00) + (N01 * N10))
        if method == "double-fault":
            diversity = N00 / (N11 + N01 + N10 + N00)
        if method == "p":
            diversity = (N11 * N00 - N01 * N10) / np.sqrt(
                (N11 + N10) * (N01 + N00) * (N11 + N01) * (N10 + N00)
            )
        if method == "disagreement":
            diversity = (N01 + N10) / (N11 + N10 + N01 + N00)

        return diversity

    def diversity_matrix(self, classifier_pool, X, Y, method="double-fault"):
        """Generates diversity matrix, of size len(classifier_pool) x len(classifier_pool) with None on diagonal

        Arguments:
            classifier_pool {np.array} -- array of estimators that support predict function
            X {np.array} -- samples on which diversity will be calculated
            Y {np.array} -- corresponding classes

        Keyword Arguments:
            method {str} -- method passed on to calculate_pair_diversity function (default: {"double-fault"})

        Returns:
            np.array -- Return a matrix of diversities between every classifier
        """
        # initialize matrix of size len(classifier_pool) x len(classifier_pool) filled with None
        div_matrix = [
            [None] * len(classifier_pool) for i in range(len(classifier_pool))
        ]
        # iterate over classifiers
        for i, cls in enumerate(classifier_pool):
            # iterate over every other classifier
            for j, other_cls in enumerate(classifier_pool):
                if cls is not other_cls:
                    # calculate diversity only above diagonal because pair diversity
                    # between 1 and 2 is equal to diversity between 2 and 1
                    if div_matrix[i][j] == None or div_matrix[j][i] == None:
                        res = self.calculate_pair_diversity(
                            cls, other_cls, X, Y, method
                        )
                        # fill upper and lower half of matrix
                        div_matrix[i][j] = res
                        div_matrix[j][i] = res
        return div_matrix

    def div_matrix_to_divs(self, div_matrix):
        """helper function to convert diversity matrix to array of mean diversities

        Arguments:
            div_matrix {np.array} -- matrix of diversities

        Returns:
            list -- returns list of mean diversities
        """
        divs = []
        for row in np.array(div_matrix):
            # calculate mean diversity between classifier and every other classifier
            divs.append(row[row != None].mean())
        return divs

    def create_final_ensemble(self, samples, neighbors=None):
        """Function creates the final ensemble of classifiers. They will later be used to predict
        the class of given samples. First calculate knn, then select N most accurate ones, from them
        select J most diverse ones.

        Arguments:
            samples {np.array} -- array of samples based on which ensamble is created 

        Keyword Arguments:
            neighbors {int} -- amount of neighbors, passed to knn algorithm (default: {None})

        Returns:
            list(BaseEstimator) -- returns list of N most accurate and J most diverse base estimators
        """
        if neighbors is None:
            neighbors = self.k
        # knn
        x, y = self.find_k_nn(samples, neighbors)
        # create list of accuracies
        accuracies = []
        for estimator in self.estimators_:
            # calcute accuracy of every estimator
            accuracies.append(self.calculate_accuracy(estimator, x, y))
        # find indices of N most accurate ones
        if type(self.N) is int:
            amount = max(1, int(-self.N))
            ind = np.argpartition(accuracies, amount)[amount:]
        elif type(self.N) is float:
            amount = max(1, int(-self.N * self.n_estimators))
            ind = np.argpartition(accuracies, amount)[amount:]
        # create an ensemble of N most accurate classifiers
        ensemble_ = []
        for index in ind:
            ensemble_.append(self.estimators_[index])
        # calculate diversity
        div_matrix = self.diversity_matrix(ensemble_, x, y)
        # convert matrix to list of diversities
        diversities = self.div_matrix_to_divs(div_matrix)
        # find indices of J most diverse ones

        if type(self.J) is int:
            amount = max(1, int(-self.J))
            ind = np.argpartition(diversities, amount)[amount:]

        elif type(self.J) is float:
            amount = int(-self.J * (self.n_estimators))
            ind = np.argpartition(diversities, amount)[amount:]

        # create final ensemble
        final_ensemble = []
        for index in ind:
            final_ensemble.append(ensemble_[index])
        return final_ensemble

    def predict(self, samples):
        """Predicts classes of given samples

        Arguments:
            samples {np.array} -- array of samples to predict

        Returns:
            list -- returns list of predicted classes
        """
        # create classes list to return
        classes = []
        # iterate over samples
        for sample in samples:
            # for every one create an ensemble of classifiers
            self.ensemble_ = self.create_final_ensemble(sample)
            # predict with every classifier from ensemble
            predicts = []
            for clas in self.ensemble_:
                predicts.append((clas.predict(sample.reshape(1, -1))).tolist())
            # find most common occuring class from predictions
            counter = sum(predicts, [])
            classes.append(max(set(list(counter)), key=counter.count))

        return np.array(classes)
