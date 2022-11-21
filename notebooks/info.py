import cv2 as cv
import numpy as np
from pathlib import Path
import pandas as pd

subnotebooks = Path.cwd()
notebooks_path = subnotebooks.parent
repo_path = notebooks_path.parent

class path_label():
    """Class to access paths and labels from csv
    """
    def __init__(self, meta=pd.read_csv(str(repo_path) + '/data/metadata.csv', sep=',')) -> None:
        self.meta = meta
        #self.paths_SET = list(meta.path_SET)
        self.pat_num = list(meta.pat_num)
        self.len = len(self.pat_num)
    
    def seg(self, rad, time, stype):
        """get path of segmentation given the radiologist, the time and the segmentation type

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
    
    def path(self, sequence, t = 't1'):
        im_seq = f'path_{sequence}_{t}' if sequence=='CMC' else f'path_{sequence}'
        paths = getattr(self.meta, im_seq)
        return list(paths)

        
# #create class to call patient and its information
# class patient():
#     """Class to access patient information
#     """
#     def __init__(self, info = path_label() ,num=0, NH = False) -> None:
#         self.path = info.paths[num] if NH == False else info.paths_NH[num]
#         self.label = info.labels[num]
#         self.FOV_x1 = info.FOV_x1[num]
#         self.FOV_x2 = info.FOV_x2[num]
#         self.FOV_y1 = info.FOV_y1[num]
#         self.FOV_y2 = info.FOV_y2[num]
#         self.NH_status = NH
#     def RGB_im(self):
#         self.image = cv.imread(str(repo_path) +"/"+ self.path)
#         self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
#         self.image = self.image[self.FOV_x1:self.FOV_x2, self.FOV_y1:self.FOV_y2] if self.NH_status == False else self.image
#         return self.image
        
#     def HSV_im(self):
#         self.HSV = cv.imread(str(repo_path) +"/"+ self.path)
#         self.HSV = cv.cvtColor(self.HSV, cv.COLOR_BGR2HSV)
#         self.HSV = self.HSV[self.FOV_x1:self.FOV_x2, self.FOV_y1:self.FOV_y2] if self.NH_status == False else self.HSV
#         return self.HSV
    
#     def gray_im(self):
#         self.gray = cv.imread(str(repo_path) +"/"+ self.path)
#         self.gray = cv.cvtColor(self.gray, cv.COLOR_BGR2GRAY)
#         self.gray = self.gray[self.FOV_x1:self.FOV_x2, self.FOV_y1:self.FOV_y2] if self.NH_status == False else self.gray
#         return self.gray
    
#     def ycrcb_im(self):
#         self.ycrcb = cv.imread(str(repo_path) +"/"+ self.path)
#         self.ycrcb = cv.cvtColor(self.ycrcb, cv.COLOR_BGR2YCrCb)
#         self.ycrcb = self.ycrcb[self.FOV_x1:self.FOV_x2, self.FOV_y1:self.FOV_y2] if self.NH_status == False else self.ycrcb
#         return self.ycrcb