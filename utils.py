# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import pandas as pd
import SimpleITK as sitk
import numpy as np

from radiomics import featureextractor
import collections


class dataset_INCan():
    """Class to access general info (paths and labels) from csv of ALL PATIENTS
    """
    def __init__(self, meta_path:Path = repo_path / 'data/dataset_info.csv') -> None:
        """dataset object for INCan dataset

        Args:
            meta_path (Path, optional): csv file path to create the object. Defaults to repo_path/'data/dataset_info.csv'.
        """
        self.meta = pd.read_csv(str(meta_path), index_col='pat_num', dtype={'pat_num':str})
        self.pat_num = list(self.meta.index)
        self.len = len(self.pat_num)

    def __repr__(self) -> str:
        return f'dataset object with {self.len} patients'
    
    def im_path(self, sequence: str, t = 't1'):
        """gets the image paths of the current meta dataframe

        Args:
            sequence (str): SMC, CMC or SET
            t (str, optional): Only for CMC (and SET). Defaults to 't1'.

        Returns:
            list: list with the paths of the images
        """
        # sequence definition
        im_seq = f'{sequence}-{t}_path' if sequence=='CMC' else f'{sequence}_path' if sequence=='SET' or sequence=='SMC' else None
        if im_seq is None:
            raise ValueError(f'not a correct sequence. Choose between SMC, CMC or SET. You chose {sequence}')
        paths = getattr(self.meta, im_seq)
        
        return list(paths)

    def seg_path(self, rad: str, time: int, stype: str='G'):
        """get paths list of segmentation given the radiologist, the time and the segmentation type

        Args:
            rad (str): Lily, Vyanka or Martha
            time (str): 1 or 2
            stype (str): global or focal

        Returns:
            list: paths of segmentation
        """
        seg_name = f'{rad}_{time}_{stype}'
        paths = getattr(self.meta, seg_name)
        return list(paths)
    
    def labels_list(self, receptor: str):
        """get labels of the patients
        Args:
            receptor (str): 'ER', 'PR' or 'HER2'

        Returns:
            list: list with labels of the patients
        """
        labels = getattr(self.meta, receptor)
        return list(labels)
    
class patient(dataset_INCan): #inherit from path_label path and seg functions
    """Class to access patient information
    """
    def __init__(self, pat_num:str, dataset = dataset_INCan()) -> None:
        """use metadata from info for each patient

        Args:
            info (obj, optional): info object. Defaults to path_label().
            num (int, optional): index of the patient (not patient number) Starts from 0. Defaults to 0.
        """
        self.meta = dataset.meta.loc[[pat_num]]
        self.pat_num = pat_num
    
    def __repr__(self) -> str:
        return f'patient object of number {self.pat_num}'
    
    def get_im(self, sequence: str, t = 't1', format:str = 'np', SET_corrected: bool = True):
        """get the image array of the patient

        Args:
            sequence (str): SMC, CMC or SET
            t (str, optional): Only for CMC (and SET). Defaults to 't1'.

        Returns:
            numpy array: image array
        """
        im_path = repo_path / self.im_path(sequence, t)[0]
        im_sitk = sitk.ReadImage(str(im_path))
        if format =='sitk':
            if sequence=='SET' and SET_corrected:
                im_array = sitk.GetArrayFromImage(im_sitk)
                im_array = im_array.astype(np.int32)
                im_array = (im_array - np.power(2,15)).astype(np.int16)
                im_sitk_corr = sitk.GetImageFromArray(im_array)
                # copy info from original image
                im_sitk_corr.CopyInformation(im_sitk)
                return im_sitk_corr
            return im_sitk
        elif format =='np':
            im_array = sitk.GetArrayFromImage(im_sitk)
            if sequence == 'SET' and SET_corrected:
                im_array = im_array.astype(np.int32)
                im_array = (im_array - np.power(2,15)).astype(np.int16)
            return im_array
        else:
            raise ValueError(f'format {format} not recognized. Choose between sitk and np')
    
    def get_seg(self, rad: str, time: int, stype: str='G', format:str = 'np'):
        """get the segmentation array of the patient

        Args:
            rad (str): Lily, Vyanka or Martha
            time (str): 1 or 2
            stype (str): global or focal

        Returns:
            numpy array: segmentation array
        """
        seg_path = repo_path / self.seg_path(rad, time, stype)[0]
        seg_sitk = sitk.ReadImage(str(seg_path))
        if format =='sitk':
            return seg_sitk
        seg_array = sitk.GetArrayFromImage(seg_sitk)
        return seg_array
    
    def get_label(self, receptor: str):
        """get label of the patient

        Args:
            receptor (str): 'ER', 'PR' or 'HER2'

        Returns:
            int: label of the patient
        """
        label = getattr(self.meta, receptor)[0]
        return label


def extractor_settings(param_path:Path, show=False):
    """set extraction settings for pyradiomics from a parameter file

    Args:
        param_path (Path or str): relative path of parameter file
        show (bool, optional): if printing setting or not. Defaults to False.

    Returns:
        obj: extractor of pyradiomics
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(str(param_path))
    if show:
        print('Extraction parameters:\n\t', extractor.settings)
        print('Enabled filters:\n\t', extractor.enabledImagetypes)
        print('Enabled features:\n\t', extractor.enabledFeatures)
    return extractor

def features_df(result:collections.OrderedDict, id_num:str):
    """given a result from pyradiomics, return a dataframe with the features

    Args:
        result (collections.OrderedDict): output of extract of pyradiomics
        id_num (str): id_number of the patient

    Returns:
        df: pd.DataFrame with the features
    """
    # get features and store in dataframe
    fv_len = 102 # number of features
    column_names = list(result.keys())[-fv_len:]
    column_names = [x.replace('original_','') for x in column_names] #remove original_ string
    df = pd.DataFrame(columns=column_names)
    # define index name
    df.index.name = 'pat_num'
    #add feature vector to df
    feature_vector = list(result.values())[-fv_len:] #get feature vector
    # set efature vector in corresponding index
    df.loc[id_num] = feature_vector

    return df