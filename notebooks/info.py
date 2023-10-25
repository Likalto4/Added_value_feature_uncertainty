from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import cv2 as cv
import SimpleITK as sitk
import pandas as pd

class path_label():
    """Class to access general info (paths and labels) from csv of ALL PATIENTS
    """
    def __init__(self, meta=pd.read_csv(str(repo_path) + '/data/metadata.csv', sep=',')) -> None:
        """only need the metadata csv file to define the object

        Args:
            meta (pd.Dataframe, optional): dataframe with the metadata. Defaults to pd.read_csv(str(repo_path) + '/data/metadata.csv', sep=',').
        """
        self.meta = meta
        self.pat_num = list(meta.pat_num)
        self.len = len(self.pat_num)

    def __repr__(self) -> str:
        return f'path_label object with {self.len} patients'
    
    def seg_path(self, rad: str, time: int, stype: str):
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
    
    def im_path(self, sequence: str, t = 't1'):
        """gets the image paths of the current meta dataframe

        Args:
            sequence (str): SMC, CMC or SET
            t (str, optional): Only for CMC (and SET). Defaults to 't1'.

        Returns:
            list: list with the paths of the images
        """
        im_seq = f'path_{sequence}_{t}' if sequence=='CMC' else f'path_{sequence}'
        paths = getattr(self.meta, im_seq)
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
      
#create class to call patient and its information
class patient(path_label): #inherit from path_label path and seg functions
    """Class to access patient information
    """
    def __init__(self, info = path_label(), num=0) -> None:
        """use metadata from info for each patient

        Args:
            info (obj, optional): info object. Defaults to path_label().
            num (int, optional): index of the patient (not patient number) Starts from 0. Defaults to 0.
        """
        self.meta = info.meta.iloc[[num]] #get metadata of patient by absolute index
        self.pat_num = self.meta.pat_num.values[0] #get patient number

    def __repr__(self) -> str:
        return f'patient {self.pat_num}'
        
    def im_array(self, sequence, t = 't1'):
        """get image as array, given the sequence and time

        Args:
            sequence (str): SMC, CMC or SET
            t (str, optional): time of the sequence (t1, t2, etc.). Defaults to 't1'.

        Returns:
            array: image array
        """
        path = self.im_path(sequence, t)
        im = cv.imread(str(repo_path) + '/' + path[0], cv.IMREAD_UNCHANGED) #keep image original format unchanged
        return im
    def im_sitk(self, sequence, t = 't1'):
        """get image as sitk image

        Args:
            sequence (str): SMC, CMC or SET
            t (str, optional): time of the sequence. Defaults to 't1'.

        Returns:
            sitk obkect: sitk image object
        """
        path = self.im_path(sequence, t)
        im = sitk.ReadImage(str(repo_path) + '/' + path[0])
        return im
    def label(self, receptor: str):
        """get label of the patient

        Args:
            receptor (str): 'ER', 'PR' or 'HER2'

        Returns:
            int: label of the patient
        """
        label = getattr(self.meta, receptor)
        return label.values[0]