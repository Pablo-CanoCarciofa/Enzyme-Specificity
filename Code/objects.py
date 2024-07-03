# General
import os
from os.path import join

# Data wrangling
import pandas as pd
import numpy as np

# Prediction
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve, confusion_matrix

# Hyperparameter Optimisation
import optuna

# Visualisation
import matplotlib.pyplot as plt

class Dataset:
    """
    Class representing a dataset
    """
    def __init__(self, df):
        """
        Initialiser method (constructor) of dataset object. Takes parameters:

        (1) df - dataframe holding data
        """
        self.df = df
    def combine_vectors(self, row, col1, col2):
        """
        Method to concatenate 2 vectors in 2 columns to one larger vector in one column in a dataframe

        (1) row - row of dataframe
        (2) col1 - column of dataframe containing first vector to be unpacked
        (3) col2 - column fo dataframe containing second vector to be unpacked
        """
        return np.concatenate([row[col1], row[col2]])
    def featurise_df(self, col):
        return np.stack(self.df[col].values)
    def featurise_train(self, df, col1, col2):
        """
        Method to create a numpy array of 2 concatenated vectors from a dataframe where each value represents
        a row in the dataframe

        (1) row - row of dataframe
        (2) col1 - column of dataframe containing first vector to be unpacked
        (3) col2 - column fo dataframe containing second vector to be unpacked
        """
        df.loc[:, (col1 + '_' + col2 + '_Combined')] = df.apply(self.combine_vectors, args = (col1, col2), axis=1)
        featurised = np.stack(df[col1 + '_' + col2 + '_Combined'].values)
        return featurised
    def train_test(self, enzyme_rep, substrate_rep, train_size):
        """
        Train test split method which creates a train test split of the dataset
        ensuring that no enzyme is split across training and testing. Takes parameters:

        (1) enzyme_rep - column containing vector with enzyme representation
        (2) substrate_rep - column containing vector with substrate representation
        (3) train_size - percentage of dataset going to training
        """
        self.enzyme_rep = enzyme_rep
        self.substrate_rep = substrate_rep
        self.train_size = train_size
        
        train_enzymes = np.random.choice(self.df['enzyme'].unique(), size=int(len(self.df['enzyme'].unique()) * self.train_size), replace=False)
        train_df = self.df[self.df['enzyme'].isin(train_enzymes)].copy()
        test_df = self.df[~self.df['enzyme'].isin(train_enzymes)].copy()

        X_train = self.featurise_train(train_df, enzyme_rep, substrate_rep)
        X_test = self.featurise_train(test_df, enzyme_rep, substrate_rep)
        y_train = train_df['active']
        y_test = test_df['active']
        return X_train, y_train, X_test, y_test
    
class Clustering:
    def __init__(self, df):
        self.df = df
    def cluster(self):
        pass
    def pca_visualisation(self):
        pass
    
class Model:
    """
    Class representing a machine learning model
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialiser method (constructor) of dataset object. Takes parameters:
        (1) X_train - training features
        (2) y_train - training labels
        (3) X_test - testing features
        (4) y_test - testing labels
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.roc_curves = []
        self.model_name = None

    def tune_hyperparameters(self, model_name, n_trials=50, verbose = False):
        """
        Hyperparameter tuning method using Optuna
        """
        self.model_name = model_name
        def objective(trial):
            if model_name == 'xgb':
                param = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                }
                model = XGBClassifier(**param)
            elif model_name == 'lr':
                param = {
                    'C': trial.suggest_float('C', 0.01, 10),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear']),
                }
                model = LogisticRegression(**param)
            elif model_name == 'svm':
                param = {
                    'C': trial.suggest_float('C', 0.01, 10),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                }
                model = SVC(**param)
            elif model_name == 'knn':
                param = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'algorithm': trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute', 'auto']),
                }
                model = KNeighborsClassifier(**param)
        
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            auroc = roc_auc_score(self.y_test, y_pred)

            fpr, tpr, _ = roc_curve(self.y_test, y_pred)
            self.roc_curves.append((fpr, tpr, auroc))

            return auroc
        
        if not verbose:
            # Silence Optuna output
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best hyperparameters: {study.best_params}")
        self.best_params = study.best_params

    def evaluate(self, model_name='xgb', tune=False, n_trials=50, verbose = False):
        """
        Model evaluation method - outputs accuracy, area under ROC curve, and ROC curve
        """  
        model_class = {
            'xgb': XGBClassifier,
            'lr': LogisticRegression,
            'svm': SVC,
            'knn': KNeighborsClassifier}.get(model_name)
        if not model_class:
            raise ValueError(f"Unsupported model: {model_name}")
            
        if tune:
            self.tune_hyperparameters(model_name, n_trials=n_trials, verbose = verbose)
            self.model = model_class(**self.best_params)
        else:
            self.model = model_class()
            
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        area_under_roc = roc_auc_score(self.y_test, y_pred)
        confuse = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Area Under ROC Curve: {area_under_roc}")
        print('Confusion Matrix:\n', confuse)

        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred, pos_label=1)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc_score(self.y_test, y_pred),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def plot_learning_curve(self):
        """
        Plots the learning curve for the given model without cross-validation
        """
        #train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_train, self.y_train)
        display = LearningCurveDisplay.from_estimator(self.model, np.concatenate([self.X_train, self.X_test]), np.concatenate([self.y_train, self.y_test]))
        display.plot()
        plt.show()
    
    def plot_roc_comparison(self):
        """
        Plots ROC curve comparison across different hyperparameters
        """
        plt.figure()

        for i, (fpr, tpr, accuracy) in enumerate(self.roc_curves):
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Trial {i+1} (Acc: {accuracy:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.show()