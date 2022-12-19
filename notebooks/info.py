import cv2 as cv
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import numpy as np

subnotebooks = Path.cwd()
notebooks_path = subnotebooks.parent
repo_path = notebooks_path.parent

class path_label():
    """Class to access generla info (paths and labels) from csv of ALL PATIENTS
    """
    def __init__(self, meta=pd.read_csv(str(repo_path) + '/data/metadata.csv', sep=',')) -> None:
        self.meta = meta
        self.pat_num = list(meta.pat_num)
        self.len = len(self.pat_num)
    
    def seg_path(self, rad, time, stype):
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
    
    def im_path(self, sequence, t = 't1'):
        im_seq = f'path_{sequence}_{t}' if sequence=='CMC' else f'path_{sequence}'
        paths = getattr(self.meta, im_seq)
        return list(paths)
      
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