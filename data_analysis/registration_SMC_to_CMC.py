#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import SimpleITK as sitk
from tqdm import tqdm
from IPython.display import clear_output
import shutil

from utils import patient, dataset_INCan

def registrater_affine(fixed:sitk.Image, moving:sitk.Image, show=False):
    """registration of two sitk images using affine transformation

    Args:
        fixed (sitk.Image): fixed image. In CEDM SET, it is the image with contrast enhancement
        moving (sitk.Image): moving image, in CEDM is the image before contrast injection

    Returns:
        sitk.ParameterMap: transform parameters
    """
    # register
    parameterMap = sitk.GetDefaultParameterMap('affine')
    parameterMap['NumberOfSpatialSamples'] = ['5000']
    #run registration
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()
    #save transform parameters
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    clear_output(wait=show)

    return transformParameterMap

def main():
    dataset = dataset_INCan()
    counter = tqdm(dataset.pat_num, total=dataset.len, desc='registration of SMC to CMC')
    for pat_num in counter:
        # read data
        pat = patient(pat_num=pat_num)
        #define fixed and moving
        fixedImage = pat.get_im(sequence='CMC', format = 'sitk')
        movingImage = pat.get_im(sequence='SMC', format='sitk')

        _ = registrater_affine(fixedImage, movingImage, show=False)
        # move created files radiomics/TransformParameters.0.txt to data
        shutil.move('TransformParameters.0.txt', repo_path / f'data/registration/transform_{pat_num}.txt')

if __name__ == '__main__':
    main()