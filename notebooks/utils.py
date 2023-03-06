from pathlib import Path
import numpy as np
import SimpleITK as sitk

subnotebooks = Path.cwd()
notebooks_path = subnotebooks.parent
repo_path = notebooks_path.parent

def min_max(im_array, show=False):
    """Get min and max values of an image

    Args:
        im_array (np.array): image array
        show (bool, optional): show min and max values. Defaults to False.

    Returns:
        tuple: min and max values
    """
    min_val = np.min(im_array)
    max_val = np.max(im_array)
    if show:
        print(f'Min: {min_val}, Max: {max_val}')
    return min_val, max_val

def save_as_nifti(array, filename, reference_path):
    """Save array as nifti image

    Args:
        array (array): array to be saved
        filename (str): path to save
        reference_image (str): path of reference image
    """
    reference_image = sitk.ReadImage(reference_path)
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(reference_image.GetOrigin())
    image.SetSpacing(reference_image.GetSpacing())
    image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(image, filename)

def GetArrayFromPath(im_path):
    im_sitk = sitk.ReadImage(str(im_path))
    im_np = sitk.GetArrayFromImage(im_sitk)
    return im_np