'''
Author: Ramin Anushiravani
Date: April 12th/23
Model Evaluation Helper
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, f1_score
import numpy as np
from sklearn import preprocessing
from .data_utils import DataUtil
import matplotlib.pyplot as plt
from IPython.display import clear_output


class ModelUtil(DataUtil):
    '''
    This class contains methods for evaluating models and keeps track of hyperparameters. Inherits from the data utility class
    '''

    def __init__(self, verbose=0):
        super().__init__()
        self.hyperparameter_search = {}
        self.verbose = verbose

    def cross_validation(self, clf, X, y, num_folds=5):
        '''
        Performs time-ordered cross-validation on a sklearn model. Each fold
        normalizes the validation data with the scaler fitted on the training
        fold, then records the training and validation accuracy.

        Args:
            clf: sklearn model with fit/score/predict_proba methods.
            X: data matrix of shape (num_samples, num_features).
            y: label vector of shape (num_samples,).
            num_folds: number of cross-validation folds.

        Returns:
            Tuple (best training accuracy, mean validation accuracy),
            each rounded to 3 decimals.
        '''

        valid_scores = []
        train_scores = []
        Gen = self.gen_data(X, y, num_folds)
        try:
            while True:
                training_data, training_label, validation_data, validation_label = next(
                    Gen)
                norm_scaler = preprocessing.StandardScaler().fit(training_data)
                training_data = norm_scaler.transform(training_data)
                validation_data = norm_scaler.transform(validation_data)

                clf.fit(training_data, training_label)  # Train model
                train_scores.append(
                    clf.score(training_data, training_label))  # training score
                valid_scores.append(
                    clf.score(
                        validation_data,
                        validation_label))  # validation score

        except StopIteration:
            done = 1

        if self.verbose:
            print("training score %.4f validation score %.4f" % (
                round(np.max(train_scores), 3),
                round(np.mean(valid_scores), 3)))

        return round(np.max(train_scores), 3), round(np.mean(valid_scores), 3)

    def find_best_hyperparameters(self, model_name):
        '''
        Returns the hyperparameters that achieved the highest validation
        accuracy for a model, from everything stored so far.

        Args:
            model_name: name of the model to look up (e.g. "RF").
                If empty, searches across all stored models.

        Returns:
            The parameter dictionary with the best validation score.
        '''
        if model_name != '':
            hyper_search = {}
            for k in self.hyperparameter_search.keys():
                for model in self.hyperparameter_search[k]:
                    if model['Model'] == model_name:
                        hyper_search[k] = model
        else:
            hyper_search = self.hyperparameter_search

        max_score = max(hyper_search.keys())
        use_param = hyper_search[max_score]
        if self.verbose:
            print(
                "Model {} \n {} \n validation score: {}".format(
                    model_name,
                    use_param,
                    '%0.4F' % max_score))

        return use_param

    def store_params(self, param, valid_score):
        '''
        Stores a model's hyperparameters alongside its validation score for
        later lookup by find_best_hyperparameters.

        Args:
            param: parameter dictionary.
            valid_score: validation accuracy achieved with those parameters.
        '''
        if valid_score in self.hyperparameter_search.keys():
            self.hyperparameter_search[valid_score].append(param)
        else:
            self.hyperparameter_search[valid_score] = [param]

    def get_roc(self, predictions, ground_truth):
        '''
        Computes ROC curve values for binary predictions.

        Args:
            predictions: predicted probabilities of class 1.
            ground_truth: test data labels.

        Returns:
            Tuple (fpr, tpr, auc, thresholds) with the false/true positive
            rates, the area under the curve, and the decision thresholds.
        '''
        fpr, tpr, thr = roc_curve(ground_truth, predictions)
        auc = roc_auc_score(ground_truth, predictions)
        return fpr, tpr, auc, thr

    def plot_cm_roc(self, prediction_prob, ground_truth, name=''):
        '''
        Plots a confusion matrix and an ROC curve side by side, and prints
        the F1 score, ROC-AUC, and the threshold with the best G-mean.

        Args:
            prediction_prob: predicted probabilities of class 1 (or a 2-D
                array of class probabilities).
            ground_truth: test data labels.
            name: title to use for the figures.
        '''
        _, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        if len(np.shape(prediction_prob)) > 1:
            t = prediction_prob[:, 1] > prediction_prob[:, 0]
            fpr, tpr, auc, thrs = self.get_roc(
                prediction_prob[:, 1], ground_truth)

        else:
            t = np.array(prediction_prob) > 0.5
            fpr, tpr, auc, thrs = self.get_roc(prediction_prob, ground_truth)

        cm = confusion_matrix(ground_truth, t)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[
                False, True]).plot(
            ax=axs[0])
        f1 = f1_score(ground_truth, t, average='weighted')
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)

        plt.title(name +
                  ' F1 Score {} ROC - AUC {} and best Thr {}'.format('%0.4F' %
                                                                     f1, '%0.4F' %
                                                                     round(auc, 5), '%0.4F' %
                                                                     thrs[ix]))
        axs[1].plot(fpr, tpr)
        plt.show()

    def real_time_test(self, model, test_datas, test_labels, thr):
        '''
        Steps through test samples one at a time, plotting each normalized
        sample colored by whether the model's prediction matches the label.

        Args:
            model: keras model with a predict method.
            test_datas: normalized test samples.
            test_labels: ground truth labels.
            thr: decision threshold on the predicted probability.
        '''
        idx = 0
        for data, label in zip(test_datas, test_labels):
            idx += 1
            pred = model.predict(data[None,], verbose=0)
            pred = pred.squeeze() > thr
            plt.figure(figsize=(15, 5), dpi=100)
            if label != pred.squeeze():
                c = 'r'
            else:
                c = 'b'
            plt.ylim([-10, 10])
            plt.plot(data, c)
            plt.title(
                "normalized data - sample {} label {} prediction {}".format(
                    idx, label, '%0.4F' % pred))
            plt.xlabel('sensors')
            plt.show()
            clear_output(wait=True)
