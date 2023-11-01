# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import numpy as np
import SimpleITK as sitk
import pandas as pd

def create_binary_mask(seg_sitk:sitk.Image, im_sitk:sitk.Image):
    """expands a segmentation to the size of the image

    Args:
        seg_sitk (sitk.Image): mask to expand
        im_sitk (sitk.Image): image to expand to

    Returns:
        sitk.Image: segmentation expanded to the size of the image
    """
    # define images origin
    im_origin = (int(im_sitk.GetOrigin()[0]), int(im_sitk.GetOrigin()[1]))
    seg_origin = (int(seg_sitk.GetOrigin()[0]), int(seg_sitk.GetOrigin()[1]))
    # remove last dimension
    seg_array = sitk.GetArrayFromImage(seg_sitk)
    seg_array = seg_array[0] # remove z dimension
    # expand array to the left and up using the adding_lu array, careful with dimension order (sitk inversion)
    adding_lu = np.subtract(seg_origin, im_origin)
    seg_array = np.pad(seg_array, ((adding_lu[1],0), (adding_lu[0],0)), 'constant', constant_values=(0,0))
    # expand to the right and down what is left using the image size
    im_size = im_sitk.GetSize()
    adding_rd = np.subtract(im_size[::-1], seg_array.shape) # invert image size tuple
    seg_array = np.pad(seg_array, ((0,adding_rd[0]), (0,adding_rd[1])), 'constant', constant_values=(0,0))
    # check if the image and the segmentation have the same size
    assert seg_array.shape == im_sitk.GetSize()[::-1], 'The image and the segmentation have different sizes'
    # create binary mask
    binary_mask = sitk.GetImageFromArray(seg_array)
    binary_mask.CopyInformation(im_sitk)
    return binary_mask

def main():
    # read valid patients num
    valid_patients = pd.read_csv(repo_path / 'data/valid_patients.csv', header=0, dtype=str)
    image_dir = repo_path /'data/images/SET'
    seg_dir = repo_path / 'data/segmentations'
    binary_dir = repo_path / 'data/binary_masks'
    binary_dir.mkdir(parents=True, exist_ok=True)

    for id_num in valid_patients['pat_num']:
        # read image
        im_path = image_dir / f'Pat_{id_num}_SET_SMC_to_CMC_1min.tif'
        im_sitk = sitk.ReadImage(str(im_path))
        # read segmentations
        for rad in ['L', 'M','V']:
            for time in ['1','2']:        
                seg_path = seg_dir / f'{rad}_{time}_seg/{id_num}_G_{rad}.seg.nrrd'
                seg_sitk = sitk.ReadImage(str(seg_path))
                # get mask and save
                binary_sitk = create_binary_mask(seg_sitk, im_sitk)
                save_dir = binary_dir / f'{rad}_{time}_seg/{id_num}_G_{rad}.seg.nrrd'
                save_dir.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(binary_sitk, str(save_dir))
                

if __name__ == '__main__':
    main()