#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from IPython.display import clear_output
from itertools import product
import radiomics

from utils import patient, dataset_INCan, extractor_settings, features_df


def transformation(im: sitk.Image, parameterMap_path:str, format='np', show=False):
    """apply transformation to an image

    Args:
        im (sitk.Image): image to be transformed
        parameterMap (sitk.ParameterMap): transformation settings
        format (str, optional): output format. Defaults to 'np'.

    Returns:
        np.array or sitk.Image: transformed image
    """
    transformixImageFilter = sitk.TransformixImageFilter()
    parameterMap = sitk.ReadParameterFile(str(parameterMap_path))
    transformixImageFilter.SetTransformParameterMap(parameterMap)
    transformixImageFilter.SetMovingImage(im)
    transformixImageFilter.Execute()
    im_result = transformixImageFilter.GetResultImage()
    clear_output(wait=show)
    if format=='sitk':
        return im_result
    elif format=='np':
        im_array = sitk.GetArrayFromImage(im_result)
        im_array = im_array.astype(np.uint16)
        return im_array
    else:
        raise ValueError('format must be "sitk" or "np"')
    

def points_in_circle(radius):
    """iterable that yields all possible combinations of x and y coordinates within cicle of fixed radius

    Args:
        radius (int): radius of circle

    Yields:
        set: set with combinations
    """
    for x, y in product(range(int(radius) + 1), repeat=2):
        if x**2 + y**2 <= radius**2:
            yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))

def main():
    # extraction settings
    param_path = repo_path / 'data/param_files/Param_2D_sym.json' #path of parameter file
    extractor = extractor_settings(param_path, show=False)
    logger = radiomics.logging.getLogger('radiomics')
    logger.setLevel(radiomics.logging.ERROR)

    # other settings
    dataset = dataset_INCan()
    saving_dir = repo_path / 'data/budget/substractions'
    saving_dir.mkdir(exist_ok=True)

    #Get all point coordinates inside the circle
    coord_list = list(points_in_circle(radius=2))

    for (x_trans, y_trans) in tqdm(coord_list):
    
        for rad in ['L','M', 'V']:
        
            for time in ['1', '2']:

                df_all = pd.DataFrame()
                # loop 3
                counter_pat = tqdm(dataset.pat_num, desc='pat_num')
                for pat_num in counter_pat:
                    #define fixed and moving
                    pat = patient(pat_num=pat_num)
                    fixedImage = pat.get_im(sequence='CMC', format = 'sitk')
                    movingImage = pat.get_im(sequence='SMC', format='sitk')
                    # transform
                    im_transformed = transformation(movingImage, repo_path / f'data/registration/transform_{pat_num}.txt', format='sitk')
                    # substract
                    fixed_array = sitk.GetArrayFromImage(fixedImage)
                    im_transformed_array = sitk.GetArrayFromImage(im_transformed)
                    im_substraction = -(fixed_array.astype(np.int32) - im_transformed_array.astype(np.int32)).astype(np.int16)
                    sitk_substraction = sitk.GetImageFromArray(im_substraction)
                    sitk_substraction.CopyInformation(fixedImage)

                    # extract features
                    result = extractor.execute(sitk_substraction, pat.get_seg(rad=rad, time=time, format='sitk'))

                    # get features and store in dataframe
                    df = features_df(result, id_num=pat_num)

                    # stack to df_all
                    df_all = pd.concat([df_all, df])
                # save
                df_all.to_csv(saving_dir / f'x{x_trans}-y{y_trans}_{rad}_{time}.csv')

if __name__ == '__main__':
    main()