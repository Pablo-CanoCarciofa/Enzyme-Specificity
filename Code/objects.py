# Objects script
# Create objects to then be called in different notebooks, to de-clutter notebooks and reduce duplication of code

### Import packages

# Data wrangling
import pandas as pd
import numpy as np
from itertools import product
from scipy.spatial.distance import cosine
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from rdkit.DataStructs import FingerprintSimilarity
import random
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Prediction
from sklearn.model_selection import LearningCurveDisplay, StratifiedKFold, cross_val_score
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

### Define classes

class Dataset:
    """
    Class representing a dataset
    """

    def __init__(self, df):
        """
        Initialiser method (constructor) of dataset object
        
        Parameters:
        (1) df - dataframe holding data
        """
        self.df = df

    def combine_vectors(self, row, col1, col2):
        """
        Concatenates 2 vectors in 2 columns to one larger vector in one column in a dataframe

        Parameters:
        (1) row - row of dataframe
        (2) col1 - column of dataframe containing first vector to be unpacked
        (3) col2 - column fo dataframe containing second vector to be unpacked

        Returns:
        (1) combined - concatenated columns
        """
        # Concatenate columns
        combined = np.concatenate([row[col1], row[col2]])
        return combined
    
    def featurise_df(self, name, embedding):
        """
        Featurises either the enzyme or substrate embeddings, keeping the name of the
        enzyme or substrate, depending on user choice. This function is only used for EDA, and not in prediction itself

        Parameters:
        (1) name - enzyme, substrate, or both (string)
        (2) embedding - column containing embedding, this will be set to particular ESM model if user specifies both 
                        in name column (string)

        Returns:
        (1) embeddings_df - dataframe with featurised representations chosen by user
        """
        # Check that user does not want both enzyme and substrate features
        if (name != 'both'):
            df = self.df
            # Drop duplicate pairs
            df = df.drop_duplicates(subset=[name]).reset_index(drop=True)
            # take only names of either enzymes or substrates and their embeddings
            df = df[[name, embedding]]
            # Convert vector in list form to separate columns, one for each value
            embeddings_df = pd.DataFrame(df[embedding].tolist(), index=df[name])
            # Rename columns by index in embedding vector
            embeddings_df.columns = [f'embedding_{i+1}' for i in range(embeddings_df.shape[1])]
        # If user wants both enzyme and substrate features, but input specific enzyme representation
        else:
            df = self.df
            # Take only enzyme and substrate names, alongside input enzyme embedding
            df = df[['enzyme', 'substrate', embedding, 'fingerprint', 'active']]
            # Convert enzyme representation vector in list form to separate columns, one for each value
            esm_df = pd.DataFrame(df[embedding].tolist(), index=df.index).add_prefix(embedding + '_')
            # Convert substrate representation vector in list form to separate columns, one for each value
            fingerprint_df = pd.DataFrame(df['fingerprint'].tolist(), index=df.index).add_prefix('fingerprint_')
            # Concatenate enzyme and substrate representation features
            embeddings_df = pd.concat([df[['enzyme', 'substrate', 'active']], esm_df, fingerprint_df], axis=1)
        return embeddings_df
    
    def featurise_train(self, df, col1, col2):
        """
        Creates a numpy array of 2 concatenated vectors from a dataframe where each value represents
        a row in the dataframe. This is used for model pre-processing

        Parameters:
        (1) row - row of dataframe
        (2) col1 - column of dataframe containing first vector to be unpacked
        (3) col2 - column fo dataframe containing second vector to be unpacked

        Returns:
        (1) featurised - numpy array of concatenated enzyme and substrate representations, with values separated
        """
        # Concatenates ESM and ECFP vector representations into one column
        df.loc[:, (col1 + '_' + col2 + '_Combined')] = df.apply(self.combine_vectors, args = (col1, col2), axis=1)
        # Stacks the representations into separate values in a numpy array
        featurised = np.stack(df[col1 + '_' + col2 + '_Combined'].values)
        return featurised
    
    def train_test(self, enzyme_rep, substrate_rep, train_size):
        """
        Creates a train test split of the dataset
        ensuring that no enzyme is split across training and testing

        Parameters:
        (1) enzyme_rep - column containing vector with enzyme representation
        (2) substrate_rep - column containing vector with substrate representation
        (3) train_size - percentage of dataset going to training

        Creates:
        (1) X_train - training data (numpy array)
        (2) X_test - test data (numpy array)
        (3) y_train - training labels (pandas series)
        (4) y_test - test labels (pandas series)
        """
        # Set class variables
        self.enzyme_rep = enzyme_rep
        self.substrate_rep = substrate_rep
        self.train_size = train_size
        
        # From set of unique enzymes randomly select input percentage of enzymes
        train_enzymes = np.random.choice(self.df['enzyme'].unique(), size=int(len(self.df['enzyme'].unique()) * self.train_size), 
                                         replace=False)

        # Take all enzyme-substrate pairs that belong to selected training enzymes
        self.train_df = self.df[self.df['enzyme'].isin(train_enzymes)].copy()
        # Take all enzyme-substrate pairs that do not belong to selected training enzymes 
        self.test_df = self.df[~self.df['enzyme'].isin(train_enzymes)].copy()

        # Create test and training sets
        self.X_train = self.featurise_train(self.train_df, enzyme_rep, substrate_rep)
        self.X_test = self.featurise_train(self.test_df, enzyme_rep, substrate_rep)
        self.y_train = self.train_df['active']
        self.y_test = self.test_df['active']

    def create_negatives(self, sub_lower=0.0, sub_upper=1.0, enz_lower=0.0, enz_upper=1.0, num_of_negs=3):
        """
        Creates inactive enzyme-substrate pairs by, for every active enzyme substrate pair, 
        randomly sampling a given number of small molecules not listed as being a substrate 
        for the enzyme, that are similar enough to the substrate in the given pair, as defined 
        by user input bounds

        Parameters:
        (1) sub_lower - lower bound on similarity of sampled inactive small molecules and given
        active substrate
        (2) sub_upper - upper bound on similarity of sampled inactive small molecules and given
        active substrate
        (3) enz_lower - lower bound on similarity of sampled negative enzyme and given
        positive enzyme
        (4) enz_upper - upper bound on similarity of sampled negative enzyme and given
        positive enzyme
        (5) num_of_negs - number of negative pairs to sample for every given positive enzyme
        substrate pair

        Updates:
        (1) df - dataframe in Dataset object
        """
        # Create DataFrame of unique SMILES representations
        unique_smiles = self.df['smiles'].unique()
        # Instantiate fingerprint generator object
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        new_rows = []

        # Iterate through every enzyme-substrate pair and get positive pair representations
        for index, row in self.df.iterrows():
            original_smile = row['smiles']
            original_esm = row['ESM1b']
            # Instantiate empty lists for similar substrates and enzymes
            similar_substrates = []
            similar_enzymes = []

            # Iterate through all unique SMILES in the data
            for unique_smile in unique_smiles:
                # Check if the given SMILES is not the same as the given substrate, and if not calculate its
                # similarity to the given substrate
                if original_smile != unique_smile:
                    similarity = FingerprintSimilarity(mfpgen.GetFingerprint(MolFromSmiles(original_smile)), 
                                                       mfpgen.GetFingerprint(MolFromSmiles(unique_smile)))
                    # If the small molecule is similar enough given input bounds, append it to similar substrates
                    # list
                    if (similarity <= sub_upper) and (similarity >= sub_lower):
                        similar_substrates.append(unique_smile)
            
            # Iterate through all enzymes
            for unique_esm in self.df['ESM1b']:
                # As before if the ESM1b of the enzyme in question is similar enough to the positive enzyme given
                # the input bounds, add it to the list of similar enzymes
                if original_esm != unique_esm:
                    enzyme_similarity = cosine(original_esm, unique_esm)
                    if (enzyme_similarity <= enz_upper) and (enzyme_similarity >= enz_lower):
                        similar_enzymes.append(unique_esm)
            
            # Calculate all possible combinations of similar enzymes and substrates and put these in a list of tuples
            combinations = list(product(similar_enzymes, similar_substrates))

            # Iterate once for every negative to create
            for _ in range(num_of_negs):
                # Randomly sample an enzyme-substrate pair from the possible negatives and append a 1 row dataframe with
                # the negative data point to the new_rows list
                try:
                    random_negative = random.choice(combinations)
                    new_row = self.df[self.df['smiles'] == random_negative[1]].iloc[0].copy()
                    new_row['enzyme'] = row['enzyme']
                    new_row['ec_number'] = row['ec_number']
                    new_row['ESM1b'] = row['ESM1b']
                    new_row['ESM2'] = row['ESM2']
                    new_row['active'] = 0
                    new_rows.append(new_row)
                    print('Negative', str(index + 1) + '/' + str(self.df.shape[0]), new_row['substrate'], 'added for', row['substrate'], 'substrate of', row['enzyme'], 'enzyme')
                except IndexError:
                    print('No negatives for', row['substrate'], 'substrate of', row['enzyme'], 'enzyme')

        # Set target column 'active' to 1 for all enzyme-substrate pairs in the experimentally-confirmed dataset    
        self.df['active'] = 1
        # Create a new dataframe from all the negatives created, dropping any duplicates, and concatenate this with the 
        # original dataframe
        new_df = pd.DataFrame(new_rows)
        new_df = new_df.drop_duplicates(subset=['smiles', 'enzyme']).reset_index(drop=True)
        print('Added', new_df.shape[0], 'negative values to', self.df.shape[0], 'positive values for total dataset size', new_df.shape[0] + self.df.shape[0])
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    
class Model:
    """
    Class representing a machine learning model
    """
    def __init__(self, dataset):
        """
        Initialiser method (constructor) of dataset object

        Parameters:
        (1) X_train - training features
        (2) y_train - training labels
        (3) X_test - testing features
        (4) y_test - testing labels
        """
        self.X_train = dataset.X_train
        self.y_train = dataset.y_train
        self.X_test = dataset.X_test
        self.y_test = dataset.y_test
        self.test_df = dataset.test_df
        self.model = None
        self.roc_curves = []
        self.model_name = None

    def tune_hyperparameters(self, model_name, n_trials=50, verbose = False):
        """
        Hyperparameter tuning using Optuna

        Parameters:
        (1) model_name - model to be fitted
        (2) n_trials - number of rounds of hyperparaneter tuning
        (3) verbose - whether to output summaries of trials after every trial

        Returns:
        (1) best_params - best parameters found during tuning
        """
        # Set class variable
        self.model_name = model_name

        # Define nested function used for iterative bayesian hyperparameter tuning
        def objective(trial):
            """
            Objective of hyperparameter tuning, sets all internal variables to tune

            Parameters:
            (1) trial - current trial in hyperparameter tuning pipeline

            Returns:
            (1) auroc - area under the receiver operating characteristic (ROC) curve (float) as
                        chosen performance metric
            """
            # Checks different models and sets out hyperparameters to be tuned with suggested ranges
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

            # Fit the model given the parameters
            model.fit(self.X_train, self.y_train)

            # Get probability predictions, for plotting ROC curve
            if hasattr(model, 'predict_proba'):
                y_pred_prob = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_prob = model.decision_function(self.X_test)
            else:
                raise ValueError(f'Model {model_name} does not have predict_proba or decision_function method')

            # Get false positive rate, true positive rate and AUROC score, and append these
            # to roc_curves class variable for visualisation
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
            auroc = roc_auc_score(self.y_test, y_pred_prob)
            self.roc_curves.append((fpr, tpr, auroc))

            return auroc
        
        # Silence Optuna output
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run bayesian hyperparameter optimisation
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Print parameters of best model
        print(f'Best hyperparameters: {study.best_params}')
        self.best_params = study.best_params

    def evaluate(self, model_name='xgb', tune=False, n_trials=50, verbose=False, cv_splits = 5):
        """
        Model evaluation method, executes hyperparameter tuning method above and outputs accuracy, 
        area under ROC curve, and ROC curve

        Parameters:
        (1) model_name - model to be fitted
        (2) n_trials - number of rounds of hyperparaneter tuning
        (3) verbose - whether to output summaries of trials after every trial

        Prints:
        (1) ROC plot - plot of tradeoff between true positives and false positives in the model
        (2) accuracy - accuracy of model
        (3) AUROC score - area under ROC curve
        (4) confusion matrix - breakdown of true and false positives and negatives
        """
        # Instantiate model dictionary
        model_class = {
            'xgb': XGBClassifier,
            'lr': LogisticRegression,
            'svm': SVC,
            'knn': KNeighborsClassifier}.get(model_name)
        if not model_class:
            raise ValueError(f'Unsupported model: {model_name}')
        
        # Run hyperparameter tuning with given input model if user has specified, saving the best
        # model found
        if tune:
            self.tune_hyperparameters(model_name, n_trials=n_trials, verbose=verbose)
            self.model = model_class(**self.best_params)
        else:
            self.model = model_class()
        
        # Fit model
        self.model.fit(self.X_train, self.y_train)

        # Get predicted probabilities for visualisation
        if hasattr(self.model, 'predict_proba'):
            y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_pred_prob = self.model.decision_function(self.X_test)
        else:
            raise ValueError(f'Model {model_name} does not have predict_proba or decision_function method.')
        
        # Get accuracy, AUROC, and confusion matrix to then print to user
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        area_under_roc = roc_auc_score(self.y_test, y_pred_prob)
        confuse = confusion_matrix(self.y_test, y_pred)

        # Cross-Validation
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_scores_rocauc = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        cv_scores_acc = cross_val_score(self.model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        print(f"Area Under ROC Curve: {cv_scores_rocauc.mean()}")
        print(f"Accuracy: {cv_scores_acc.mean()}")
        print('Confusion Matrix:\n', confuse)

        # Get false positive and true positive rate to plot ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob, pos_label=1)
        self.roc_curves.append((fpr, tpr, area_under_roc, model_name))
        
        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw,
            label='ROC curve (area = %0.2f)' % roc_auc_score(self.y_test, y_pred_prob),
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def plot_learning_curve(self):
        """
        Plots the learning curve for the given model
        """
        display = LearningCurveDisplay.from_estimator(self.model, np.concatenate([self.X_train, self.X_test]), np.concatenate([self.y_train, self.y_test]))
        display.plot()
        plt.show()
    
    def plot_roc_comparison(self, num_curves=5, mode='spaced'):
        """
        Plots ROC curve comparison across different hyperparameters, once hyperparameter tuning
        method has been called. User can choose to show the top `num_curves` ROC curves or the 
        most spaced out `num_curves` ROC curves in terms of their AUROC values. This is for best
        visualisation in the paper
        
        Parameters:
        (1) roc_curves - list of (fpr, tpr, auroc) tuples from hyperparameter tuning process
        (2) num_curves - number of ROC curves to display
        (3) mode - 'top' to show top ROC curves or 'spaced' to show most spaced out ROC curves

        Plots:
        (1) ROC plot - ROC curve for different rounds of hyperparameter tuning on the same axes, to
                       show improvement
        """
        # Sort ROC curves by AUROC in ascending order, since it is 3rd item in tuples in roc_curves
        sorted_roc_curves = sorted(self.roc_curves, key=lambda x: x[2], ascending = False)

        # Instantiate empty list for unique ROC scores
        unique_auroc_scores = []
        # Create set to avoid duplicate ROC scores in same visualisation
        seen_aurocs = set()
        # Iterate through sorted ROC curves
        for curve in sorted_roc_curves:
            # If AUROC score rounded to 2dp has not already appeared in previous hyperparameter tuning round then
            # append to unique AUROC score list
            if round(curve[2], 2) not in seen_aurocs:
                unique_auroc_scores.append(curve)
                seen_aurocs.add(round(curve[2], 2))

        # If user wants to see top ROC curves then just select highest AUROC scores
        if mode == 'top':
            selected_roc_curves = unique_auroc_scores[-num_curves:]

        # If user wants to see most spaced out ROC curves then first check if the number they want to see
        # is less than the number of tuning rounds, if so then you can just show them all ROC curves
        elif mode == 'spaced':
            num_all_curves = len(unique_auroc_scores)
            if num_all_curves <= num_curves:
                selected_roc_curves = unique_auroc_scores
            # Otherwise, create an equally spaced list running from 0 to the number of curves we have, then
            # select these curves by index from the ROC curves we have ordered by AUROC score
            else:
                indices = np.linspace(0, num_all_curves - 1, num_curves, dtype=int)
                selected_roc_curves = [unique_auroc_scores[i] for i in indices]
        else:
            raise ValueError('Invalid mode. Choose top or spaced')
        
        # Plot the selected ROC curves
        plt.figure()
        for i, (fpr, tpr, auroc, _) in enumerate(reversed(selected_roc_curves)):
            plt.plot(fpr, tpr, lw=2, label=f'ROC {i+1} (AUROC: {auroc:.2f})')
    
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.show()
    
    def get_wrongly_classified(self):
        """
        Returns a DataFrame of all the wrongly classified rows after model evaluation

        Returns:
        (1) filtered_df - original dataframe but with true label and predicted label columns appended
        """
        # Check a model has been trained
        if self.model is None:
            raise ValueError('The model has not been trained or evaluated yet')
        
        # Get predictions on the test set and all of the indices of the misclassifications
        y_pred = self.model.predict(self.X_test)
        misclassified_indices = self.y_test != y_pred
        # Create a numpy array that is the same length as the test array, with a 1 if a sample has been
        # misclassified and 0 if not
        test_length = len(self.test_df)
        misclassified_length = len(misclassified_indices)
        padded_misclassified = np.pad(misclassified_indices, (0, test_length - misclassified_length), constant_values = False)
        # Filter the test set by this numpy array and create new columns for true and predicted labels
        filtered_df = self.test_df[padded_misclassified]
        filtered_df = filtered_df.rename(columns = {'active' : 'true_label'})
        filtered_df['predicted_label'] = 1 - filtered_df['true_label']
        return filtered_df
    
    def predict(self, X):
        """
        Predict the label and probability for the given input X

        Parameters:
        (1) X - input array for which the prediction is to be made

        Returns:
        (1) label - predicted label (integer)
        (2) probability - probability of the predicted label (float)
        """
        # Check if the model has been trained yet
        if self.model is None:
            raise ValueError('The model has not been trained yet')
        
        # Get label predictions on input data
        label = self.model.predict(X)

        # Get probability of those predictions
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X)[:, 1]  # Probability of the positive class
        else:
            # For models without predict_proba, use decision_function
            probability = self.model.decision_function(X)
            # Convert decision function output to probability using a logistic function
            probability = 1 / (1 + np.exp(-probability))
        return label, probability
    
    def compare_models(self, model_names=['xgb', 'lr', 'svm', 'knn'], tune=False, n_trials=50, verbose=False):
        """
        Compare ROC curves for multiple model types

        Parameters:
        (1) model_names - List of model names to compare
        (2) tune - Whether to perform hyperparameter tuning
        (3) n_trials - Number of trials for hyperparameter tuning
        (4) verbose - Whether to output summaries of trials after every trial

        Plots:
        (1) ROC plot - ROC plot for each model type on the same axes
        """
        # Iterate through input models and evaluate these
        for model_name in model_names:
            self.evaluate(model_name=model_name, tune=tune, n_trials=n_trials, verbose=verbose)

        # Plot ROC curve for each model on same axes
        plt.figure()
        for fpr, tpr, auroc, model_name in self.roc_curves:
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUROC: {auroc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc='lower right')
        plt.show()
    