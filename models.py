# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.7")

import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
# detectron
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Boxes
# machine learning
import sklearn
# normalize the features
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
# classifiers
from sklearn.model_selection import GridSearchCV
# use leave one out cross validation
from sklearn.model_selection import LeaveOneOut
# metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import wilcoxon
# local imports
from utils import dataset_INCan

setup_logger()

class DBT_extractor():
    """Class to extract the backbone features from an image
    """
    def __init__(self, config_file:str, model_file:str, min_score:float):
        """Initialize the class

        Args:
            config_file (str): path to the config file
            model_file (str): path to the model file
            min_score (float): minimum score to keep a prediction
        """
        self.config_file = config_file
        self.model_file = model_file
        self.min_score = min_score
        self.predictor = self._initialize_predictor()
        self.main_df = None
        self.main_df_path = None
        self.feature_name = None
        
    def _initialize_predictor(self):
        """Initialize the predictor

        Returns:
            detectron2.engine.DefaultPredictor: predictor
        """
        cfg = get_cfg()
        cfg.merge_from_file(self.config_file)
        cfg.MODEL.WEIGHTS = self.model_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.min_score  # set the testing threshold for this model
        predictor = DefaultPredictor(cfg)
        return predictor
    
    # when printing
    def __repr__(self):
        return f'DBT_extractor(config_file={self.config_file}, model_file={self.model_file}, min_score={self.min_score})'

    def get_normal_BBox (self, im_array:np.array):
        """Given an mammogram image, returns the bounding box of the breast

        Args:
            im_array (np.array): array of the mammogram image, with background black

        Returns:
            tuple, array: bounding box coordinates, and image with the breast only
        """
        #threshold im_array 
        img = cv.threshold(im_array, 0, 255, cv.THRESH_BINARY)[1]  # ensure binary
        nb_components, output, stats, _ = cv.connectedComponentsWithStats(img, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        img2 = np.zeros(output.shape,dtype=np.uint8)
        img2[output == max_label] = 255
        contours, _ = cv.findContours(img2,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        cnt = contours[0]

        x,y,w,h = cv.boundingRect(cnt)
        
        return (x,y,x+w,y+h), img2

    def prepare_bbox(self, bbox_lesion:np.array, predictor:detectron2.engine.defaults.DefaultPredictor, image_rgb:np.array):
        """Transform bbox to the format of the backbone

        Args:
            bbox_lesion (np.array): bbox in the format [x1, y1, x2, y2]
            predictor (detectron2.engine.defaults.DefaultPredictor): predictor, to know the augmentation technique
            image_rgb (np.array): image, to know the resizing

        Returns:
            Boxes: transformed bbox
        """
        # transform bbox to the format of the backbone
        new_bbox_lesion = predictor.aug.get_transform(image_rgb).apply_box([bbox_lesion])
        new_bbox_lesion = torch.as_tensor(new_bbox_lesion).cuda()
        # transform to boxes object
        new_bbox_lesion = Boxes(new_bbox_lesion)
        assert new_bbox_lesion.tensor.shape == (1, 4)
        
        return new_bbox_lesion

    def backbone_feature_extraction(self, predictor:detectron2.engine.DefaultPredictor, image_rgb:np.array):
        """Extract the backbone features from the image

        Args:
            predictor (detectron2.engine.DefaultPredictor): default predictor
            image_rgb (np.array): image to extract the features from

        Returns:
            dict: dictionary with the feature maps, p2 to p6
        """
        ### PYRAMID FEATURES
        with torch.no_grad():
            height, width = image_rgb.shape[:2]
            image = predictor.aug.get_transform(image_rgb).apply_image(image_rgb)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            images = predictor.model.preprocess_image([inputs]) # additional preprocessing step

            feature_maps = predictor.model.backbone(images.tensor)

        return feature_maps

    def bbox_pooler_feature_extraction(self, predictor:detectron2.engine.DefaultPredictor, feature_maps:dict, new_bbox_lesion:Boxes):
        """Extract the features from the bbox using the head pooler, activily using the bbox to crop the feature maps

        Args:
            predictor (detectron2.engine.DefaultPredictor): default predictor
            feature_maps (dict): original pyramid feature maps
            new_bbox_lesion (Boxes): transformed bbox of lesion

        Returns:
            torch.Tensor: tensor with the features of the bbox, 256 features of 7x7
        """
        features = [feature_maps[f] for f in predictor.model.roi_heads.box_in_features]
        box_features = predictor.model.roi_heads.box_pooler(features, [new_bbox_lesion])
        return box_features
        
    def extract_1024(self, image_rgb:np.array, bbox_lesion:np.array):
        """Extract the 1024 features from the image and bbox

        Args:
            image_rgb (np.array): image to extract the features from
            bbox_lesion (np.array): bbox of the lesion

        Returns:
            np.array: 1024 features
        """
        if self.feature_name != 'features_1024':
            self.feature_name = 'features_1024' # update feature name
        # we make the bbox match the format of the backbone
        new_bbox_lesion = self.prepare_bbox(bbox_lesion, self.predictor, image_rgb)

        # BACKBONE FF-FEATURES
        feature_maps = self.backbone_feature_extraction(self.predictor, image_rgb)

        # BBOX POOLER FEATURES
        box_features = self.bbox_pooler_feature_extraction(self.predictor, feature_maps, new_bbox_lesion)

        # HEAD FEATURES (1024)
        box_features_after_head = self.predictor.model.roi_heads.box_head(box_features)
        # send to cpu
        box_features_after_head = box_features_after_head.detach().cpu().numpy()[0]
        assert box_features_after_head.shape == (1024,)
        return box_features_after_head
    
    def features_to_csv(self, features:np.array, pat_num:str or int):
        """Transform the features to a csv file. The first column is the patient ID, the rest are the features

        Args:
            features (np.array): features to transform, as a 1D array
            pat_num (strorint): patient number

        Returns:
            pd.DataFrame: dataframe with the features
        """
        range_f = features.shape[0]
        df = pd.DataFrame(features).T
        df.insert(0, 'PatientID', pat_num)
        df.columns = ['PatientID'] + [i for i in range(1, range_f+1)]

        return df
    
    def update_main_df(self, df:pd.DataFrame):
        """Update the main dataframe with the new features

        Args:
            df (pd.DataFrame): dataframe with the features
        """
        self.main_df = pd.concat([self.main_df, df], ignore_index=True)

    def save_main_df(self, rad:str, time:str or int, save_path:Path=None):
        """Save the main dataframe to a csv file

        Args:
            rad (str): radiologist name
            time (str or int): time of the exam
        """
        assert self.main_df is not None, 'The main dataframe is empty'
        # warning if the main_df is not complete
        if len(self.main_df) < 33:
            print('WARNING: The main dataframe is not complete')
        
        self.main_df_path = repo_path / f'data/deep/features/{self.feature_name}/{rad}_{time}_features.csv' if save_path is None else save_path
        # make sure parent exists
        self.main_df_path.parent.mkdir(parents=True, exist_ok=True)
        self.main_df.to_csv(self.main_df_path, index=False)
        print(f'Main dataframe saved to {self.main_df_path}')
        # reset main_df
        self.main_df = None

# machine learning predictor class
class predictor_machine():
    def __init__(self):
        # self.feature_type = feature_type
        self.clf = None
        self.scaler = StandardScaler()
        self.cv = LeaveOneOut()
        self.original_features = None
        self.features = None
        self.num_samples = 0
        self.receptor = None
        self.labels = None
        # preprocessing
        self.scale_together = False
        self.corr_threshold = None
        # lists
        self.pos_probabilities = []
        self.true_labels = []
        self.best_estimators = []
        self.loo_scaler = []
        # budget
        # if self.feature_type=='radiomics':
        #     self.budget = pd.read_csv(repo_path / 'data/budget/budget_radiomics/budget_std.csv', index_col=0).mean(axis=0)
        # elif self.feature_type=='deep':
        #     self.budget = pd.read_csv(repo_path / 'data/budget/budget_std.csv', index_col=0).mean(axis=0)
        self.budget = pd.read_csv(repo_path / 'data/budget/combined_budget/budget_std_combined.csv', index_col=0).mean(axis=0)
        self.testing_synthetic_units = 1000
        self.training_synthetic_units = 50
        self.testing_budget_scale = 1
        self.training_budget_scale = 1
        self.X_test_augmented = None

    def set_classifier(self, pred:sklearn.base.BaseEstimator, parameters:dict, verbose:int=0):
        # grid search, the best model is selected based on the roc_auc score
        self.clf = GridSearchCV(pred, parameters, cv=5, scoring='roc_auc', verbose=verbose, n_jobs=6, return_train_score=True)

    def eliminate_highly_correlated(self, features:pd.DataFrame):
        # find zeros
        features = features.loc[:, (features != 0).any(axis=0)]
        # compute the correlation matrix
        corr = features.corr().abs()
        # get the upper triangle of the correlation matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        print(f'Features with high correlation: {to_drop}')
        # drop features
        features = features.drop(to_drop, axis=1)
        print(f'Features with high correlation dropped, {features.shape[1]} features remaining.')
        
        return features

    def select_features(self, features:pd.DataFrame, labels:pd.DataFrame, n_features:int):
        # get the best features
        self.selector = SelectKBest(f_classif, k=n_features)
        # fit the selector
        self.selector.fit(features, labels.values.ravel())
        # get the selected features
        features = features.iloc[:,self.selector.get_support()]
        self.selected_features = self.selector.get_feature_names_out()
        print(f'The selected features are: {features.columns.values}')

        return features

    def prepare_features(self, features:pd.DataFrame, show_info=True, scale_together=False, n_features=4, corr_threshold=0.99, training:bool=True):
        """with the defined scaler, scale feature and return as a dataframe

        Args:
            features (pd.DataFrame): input features, in order of prediction. Only numerical values.

        Returns:
            pd.DataFrame: scaled features
        """  
        features = features.sort_index() # ensure index is in order
        self.original_features = features.copy()
        if show_info:
            print(f'Original features, {features.shape[1]} features.')
        if training:
            # eliminate highly correlated features
            self.corr_threshold = corr_threshold
            features = self.eliminate_highly_correlated(features)
            # select the best features
            assert self.labels is not None, 'Please set the receptor first.'
            features = self.select_features(features, self.labels, n_features=n_features)
            self.scale_together = scale_together # only decide the self value if training
        else:
            # set the selected features
            print(f'Using the selected features: {self.selected_features}')
            features = features[self.selected_features]
        
        if self.scale_together:
            self.scaler.fit(features)
            features = pd.DataFrame(self.scaler.transform(features), columns=features.columns)
            print('Features scaled together. This could represent DATA LEAKAGE.') # show warning
        self.num_samples = len(features)
        if show_info:
            print(f'Features prepared, {self.num_samples} samples, {features.shape[1]} features.')
        self.features = features
        
    def set_receptor(self, receptor:str, show_distribution:bool=False):
        """set the receptor to predict, it prepare the labels format

        Args:
            receptor (str): name of the receptor to predict
            show_distribution (bool, optional): show the positive distribution. Defaults to False.
        """
        self.receptor = receptor
        labels = dataset_INCan().labels_list(receptor=self.receptor)
        self.labels = pd.DataFrame(labels, columns=[self.receptor])
        if show_distribution:
            print(f'The positive cases of {self.receptor} represent {self.labels[self.receptor].mean().round(3)*100}%')  

    def train(self):
        # restart the lists
        self.pos_probabilities = []
        self.true_labels = []
        self.best_estimators = []
        self.loo_scaler = []

        for train_index, test_index in self.cv.split(self.features):
            # get the train and test data
            X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
            # scale the data
            if not self.scale_together:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                self.loo_scaler.append(self.scaler) # save the scaler for inference
            # get the train and test labels
            y_train, y_test = self.labels.iloc[train_index], self.labels.iloc[test_index]
            # fit the model
            self.clf.fit(X_train, y_train.values.ravel())
            # save best params of this iteration
            self.best_estimators.append(self.clf.best_estimator_)
            # predict the probability
            y_prob = self.clf.predict_proba(X_test)[0,1] # get only the positive class
            # append the prediction
            self.pos_probabilities.append(y_prob)
            self.true_labels.append(y_test.values[0,0])
        # convert to arrays
        self.true_labels = np.asarray(self.true_labels)
        self.pos_probabilities = np.asanyarray(self.pos_probabilities)

        print(f'Training finished!')

    def compute_metrics(self, plot_metrics:bool=True):
        
        roc_auc = roc_auc_score(self.true_labels, self.pos_probabilities)
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.pos_probabilities)
        # get ideal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        # get optimal predictions
        optimal_predictions = np.where(self.pos_probabilities > optimal_threshold, 1, 0) # when to predict positive
        accuracy = accuracy_score(self.true_labels, optimal_predictions)
        precision = precision_score(self.true_labels, optimal_predictions)
        recall = recall_score(self.true_labels, optimal_predictions)
        f1 = f1_score(self.true_labels, optimal_predictions)
        if not plot_metrics:
            print(f'\nAUC:{roc_auc}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\n')
        if plot_metrics:
            # plot confusion matrix on the left
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            cm = confusion_matrix(self.true_labels, optimal_predictions)
            ConfusionMatrixDisplay(cm).plot(ax=ax[0])
            ax[0].set_title('Confusion Matrix')
            # plot the ROC curve on the right
            ax[1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})')
            ax[1].plot([0,1],[0,1], 'k--')
            ax[1].set_xlabel('False Positive Rate')
            ax[1].set_ylabel('True Positive Rate')
            ax[1].set_title('ROC Curve')
            ax[1].scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label=f'Best Threshold: {optimal_threshold.round(3)}')
            ax[1].legend()
            # plot the metrics on the right, using thin bars
            ax[2].barh(['Accuracy', 'Precision', 'Recall', 'F1'], [accuracy, precision, recall, f1], height=0.5)
            # set the limits
            ax[2].set_xlim(0,1)
            ax[2].set_xticks(np.arange(0,1.1,0.1))
            # write the values on the bars
            for i, v in enumerate([accuracy, precision, recall, f1]):
                ax[2].text(v+0.01, i-0.1, f'{v.round(2)}')
                
            ax[2].set_title('Metrics')
            fig.tight_layout()
            plt.show()
        return roc_auc
            
    def dumb_inference(self):
        """infere with a feature vector of zeros"""
        # restart the lists
        self.pos_probabilities = []
        self.true_labels = []
        # create a dumb feature vector
        dumb_features = pd.DataFrame(np.zeros((self.features.shape[0], self.features.shape[1])))
        # predict
        for _, test_index in self.cv.split(self.features):
            # get the train and test data
            X_test = dumb_features.iloc[test_index]
            X_test = pd.DataFrame(X_test, columns=self.features.columns) # scale the features
            y_test = self.labels.iloc[test_index]
            # predict the probability
            y_prob = self.best_estimators[test_index[0]].predict_proba(X_test)[0,1] # get only the positive class
            # append the prediction
            self.pos_probabilities.append(y_prob)
            self.true_labels.append(y_test.values[0,0]) 
        # convert to arrays
        self.true_labels = np.asarray(self.true_labels)
        self.pos_probabilities = np.asanyarray(self.pos_probabilities)
        print('Dumb prediction, using zero-valued feature vectors, NO training done.')

    def ordered_inference(self):
        # restart the lists
        self.pos_probabilities = []
        self.true_labels = []
        # predict
        for _, test_index in self.cv.split(self.features):
            # get the train and test data
            X_test = self.features.iloc[test_index]
            if not self.scale_together:
                X_test = self.loo_scaler[test_index[0]].transform(X_test)
            y_test = self.labels.iloc[test_index]
            # predict the probability
            y_prob = self.best_estimators[test_index[0]].predict_proba(X_test)[0,1] # get only the positive class
            # append the prediction
            self.pos_probabilities.append(y_prob)
            self.true_labels.append(y_test.values[0,0])
        # convert to arrays
        self.true_labels = np.asarray(self.true_labels)
        self.pos_probabilities = np.asanyarray(self.pos_probabilities)
        print('Recomputed prediction, NO training done.')

    def budget_inference(self):
        # restart the lists
        self.pos_probabilities = []
        self.true_labels = []

        # predict
        for _, test_index in self.cv.split(self.features):
            # repeat the test sample to match the number of synthetic samples
            y_test = self.labels.iloc[test_index]
            y_test_augmented = np.repeat(y_test, self.testing_synthetic_units, axis=0)

            # generate the synthetic samples
            X_test_base = self.features.iloc[test_index]
            # augment the test sample
            self.X_test_augmented = self.augment_sample(X_test_base, budget_scale=self.testing_budget_scale, training=False)

            # predict the probability
            y_prob = self.best_estimators[test_index[0]].predict_proba(self.X_test_augmented)

            # append the prediction
            self.pos_probabilities.append(y_prob)
            self.true_labels.append(y_test_augmented)
        # convert to arrays
        self.true_labels = np.asarray(self.true_labels).reshape(-1,1)[:,0]
        self.pos_probabilities = np.asanyarray(self.pos_probabilities).reshape(-1,2)[:,1]
        print('Computed prediction using budge, NO training done.')

        return self.pos_probabilities

    def augment_sample(self, X_base:pd.DataFrame, budget_scale:float=1, training:bool=False):
        """given a base test set, augment it with random samples from a gaussian distribution with mean zero and std given by the budget

        Args:
            X_base (pd.DataFrame): base of the augmentation

        Returns:
            pd.DataFrame: augmented test set
        """
        synthetic_units = self.training_synthetic_units if training else self.testing_synthetic_units
        X_base = pd.DataFrame(self.scaler.inverse_transform(X_base), columns=X_base.columns) # send back to original scale <-----
        X_base = pd.concat([X_base]*synthetic_units, ignore_index=True)
        # generate 1000 random samples from a gaussian distribution, mean zero and std given by the budget
        budget = self.budget[self.budget.index.isin(self.selected_features)]
        random_samples = np.random.normal(0, budget*budget_scale, size=(X_base.shape[0], X_base.shape[1]))
        random_samples = pd.DataFrame(random_samples, columns=X_base.columns)
        # sum the random samples to the base features
        X_augmented = X_base + random_samples
        X_augmented = pd.DataFrame(self.scaler.transform(X_augmented), columns=X_augmented.columns) # scale back according to scaler ---->

        return X_augmented
    
    def budget_training(self):
        self.best_estimators = []
        self.pos_probabilities = []
        self.true_labels = []

        for train_index, test_index in self.cv.split(self.features):
            # get the train and test data
            X_train, X_test = self.features.iloc[train_index], self.features.iloc[test_index]
            # get the train and test labels
            y_train, y_test = self.labels.iloc[train_index], self.labels.iloc[test_index]
            
            # expand the train sample to match the number of synthetic samples
            y_train_augmented = pd.concat([y_train]*self.training_synthetic_units, ignore_index=True)
            
            # invert the features to the original scale
            X_train_augmented = self.augment_sample(X_train, budget_scale=self.training_budget_scale, training=True)
            # fit the model
            self.clf.fit(X_train_augmented, y_train_augmented.values.ravel())
            # save best params of this iteration
            self.best_estimators.append(self.clf.best_estimator_)
            # predict the probability
            y_prob = self.clf.predict_proba(X_test)[0,1] # get only the positive class
            # append the prediction
            self.pos_probabilities.append(y_prob)
            self.true_labels.append(y_test.values[0,0])
        # convert to arrays
        self.true_labels = np.asarray(self.true_labels)
        self.pos_probabilities = np.asanyarray(self.pos_probabilities)

        print(f'Training with budget finished!')

    def box_plots_budget(self, pos_probabilities_trad, pos_probabilities_rob, ideal_auc, show_pvalue=False):
        # reorder the probabilities array, to be a 33x1000 array
        pos_probabilities_trad = pos_probabilities_trad.reshape(-1,self.testing_synthetic_units)
        pos_probabilities_rob = pos_probabilities_rob.reshape(-1,self.testing_synthetic_units)
        true_labels = self.true_labels.reshape(-1,self.testing_synthetic_units)

        auc_trad = []
        auc_rob = []
        for i in range(true_labels.shape[0]):
            auc_trad.append(roc_auc_score(true_labels[:,i], pos_probabilities_trad[:,i]))
            auc_rob.append(roc_auc_score(true_labels[:,i], pos_probabilities_rob[:,i]))
        # list to array
        auc_trad = np.asarray(auc_trad)
        auc_rob = np.asarray(auc_rob)

        # plot boxplot
        plt.figure()
        plt.boxplot([auc_trad, auc_rob], showmeans=True, meanline=True, showfliers=True)
        plt.xticks([1,2], ['Traditional training', 'Robust training'])
        plt.ylabel('AUC')
        plt.title(f'AUC comparison for {self.receptor}')
        # show in a point the ideal auc value
        plt.scatter(1, ideal_auc , marker='o', color='black', label='Ideal AUC')
        plt.legend()
        plt.show()
        
        # test statistical difference
        if show_pvalue:
            print(f'Wilcoxon signed-rank test p-value: {wilcoxon(auc_trad, auc_rob).pvalue}')
            if wilcoxon(auc_trad, auc_rob).pvalue > 0.05:
                print(f'This means that the difference is not significant, and the robust method is not better than the traditional method.')
            else:
                print(f'This means that the difference is significant, and the robust method is better than the traditional method.')