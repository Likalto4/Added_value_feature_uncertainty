# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

# #sklearnex
# from sklearnex import patch_sklearn
# patch_sklearn()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# StratifiedKFold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score,
                            recall_score,
                            precision_score,
                            f1_score,
                            roc_auc_score,
                            roc_curve,
                            matthews_corrcoef,
)
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
from radiomics import featureextractor
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler
# RandomizedSearchCV
#Import paths and patients classes
from notebooks.info import path_label, patient
import notebooks.utils as utils

def get_features(show=False):
    """returns available features for the uncertainty scheme

    Args:
        show (bool, optional): if information about the excluded features and nxp size should be given. Defaults to False.

    Returns:
        pd.DataFrame: features loaded
    """
    budget_path = repo_path/ 'data' / 'budget' / 'budget_ROI_and_rad.csv'
    excluded, _ = utils.get_ex_included(budget_path) # get excluded features due to their budget value
    features = pd.read_csv(repo_path / 'data' / 'features' / f'feat_vector.csv', index_col=0)
    # remove features in excluded list
    features = features.drop(excluded, axis=1)
    if show:
        print(f'The features removed are {excluded.values} beacuse their budget value is greater than 1')
        n = features.shape[0] # number of patients
        p = features.shape[1] # number of features
        print(f'The number of patients (n) is: {n}\nThe number of features (p) is: {p}')
    return features


def main():

    # experiment HP
    label = 'RP' # receptor type (RP, RE, ki67)

    # get pat info and features
    info = path_label()
    features = get_features(show=True)

    # train traditional

    y = np.asarray(info.labels_list(label))
    x = np.asarray(features)
    # define lasso
    pipe_lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LogisticRegression(penalty='l1', solver='liblinear', class_weight=None, C=10, max_iter=1000))
        ])
    # apply in ALL dataset
    pipe_lasso.fit(x, y)
    coef = pipe_lasso.named_steps['lasso'].coef_
    # get the features that are not zero
    features_selected = features.columns[coef[0] != 0]
    print(f'The number of features selected is: {len(features_selected)}')
    print(f'The features selected are: {features_selected.values}')

    # predict
    y_pred = pipe_lasso.predict_proba(x)
    # get acc
    acc = accuracy_score(y, y_pred.argmax(axis=1))
    print(f'The accuracy is: {acc}')


    x = np.asarray(features[features_selected])
    # the lasso selected features size are:
    print(f'The last size of the features is: {x.shape}')
