# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import SimpleITK as sitk
import numpy as np

from utils import dataset_INCan, patient
from scipy import stats
import cv2 as cv
from tqdm import tqdm
import pandas as pd

def prepare_array(array:np.array, min_val:int, max_val:int):
    """map array to the range 0-255, and clip values below min_val and above max_val

    Args:
        array (np.array): image array
        min_val (int): min value to clip
        max_val (int): max value to clip

    Returns:
        np.array: image array clipped and mapped to 0-255
    """
    # send to min value all piel below that value
    array[array<min_val] = min_val
    # same for max value
    array[array>max_val] = max_val
    # map im_array to 256 pixels
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = (array * 255).astype(np.uint8)
    return array

def get_normal_BBox (im_array:np.array):
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

def get_bbox_cedm(SET_array:np.array, pat_num:int or str):
    """Given an SET image, returns the bounding box of the breast

    Args:
        SET_array (np.array): SET array in original format
        pat_num (int or str): patient number

    Returns:
        tuple: bounding box coordinates
    """
    SET_mode = stats.mode(SET_array[SET_array!=0].flatten(), keepdims=True)[0][0]
    # send pixels to 0 in the mode
    SET_array_blackbg = SET_array.copy()
    SET_array_blackbg[SET_array_blackbg==SET_mode] = 0
    # also remove +-range the mode in one line
    elimination_range = 11 if pat_num in ['8', '38'] else 7
    for i in range(elimination_range): # 7 is the standard value, for special cases 11 or 15 may work. Check final bbox
        SET_array_blackbg[(SET_array_blackbg==SET_mode+i) | (SET_array_blackbg==SET_mode-i)] = 0

    bbox, _ = get_normal_BBox(SET_array_blackbg)

    return bbox

def main():
    ### Summary
    dataset_info = dataset_INCan()
    # define min and max values for the images
    min_val = -168
    max_val = 232
    stype = 'G'

    # dirs
    image_dir = repo_path / 'data/deep'/ 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir = repo_path / 'data/deep'/ 'breast_bbox'
    bbox_dir.mkdir(parents=True, exist_ok=True)

    bboxes_df = None

    # loop on patients
    for pat_num in tqdm(dataset_info.pat_num):
    # for pat_num in tqdm(['8']):
        patient_ex = patient(pat_num)
        # read SET image, left oriented and corrected
        SET_array = patient_ex.get_im(sequence='SET', format='np', SET_corrected=True, left_oriented=True)
        SET_array = prepare_array(SET_array, min_val, max_val) # to 8bits in range

        # get bbox
        bbox = get_bbox_cedm(SET_array, pat_num)

        # CROP IMAGES and MASKS
        SET_array_cropped = SET_array[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        # save image as png
        im_png = sitk.GetImageFromArray(SET_array_cropped)
        sitk.WriteImage(im_png, str(image_dir / f'Pat_{pat_num}_SET.png'))

        # save bbox coordinates
        df = pd.DataFrame(columns=['pat_num','x1','y1','x2','y2'])
        df.loc[0] = [pat_num, bbox[0], bbox[1], bbox[2], bbox[3]]
        bboxes_df = pd.concat([bboxes_df, df])

        for rad in ['L','M', 'V']:
            for time in ['1','2']:
                mask_array = patient_ex.get_seg(rad=rad, time=time, left_oriented=True, stype=stype)
                mask_array_cropped = mask_array[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                # send values of 1 to 255
                mask_array_cropped[mask_array_cropped==1] = 255
                # save mask as png
                im_png = sitk.GetImageFromArray(mask_array_cropped)
                mask_dir = repo_path / 'data/deep'/ f'{stype}_masks'
                mask_dir.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(im_png, str(mask_dir / f'Pat_{pat_num}_mask_{rad}_{time}.png'))
    # save bboxes
    bboxes_df.to_csv(str(bbox_dir / 'coords.csv'), index=False)
        
if __name__ == "__main__":
    main()