from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd

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

def get_ex_included(budget_path: Path):
    """get excluded features name using the budget CV value

    Args:
        budget_path (Path): path to the csv file with the budget

    Returns:
        sequences: excluded and included features
    """
    # get the name of the features from the budget
    budget = pd.read_csv(budget_path, index_col=0)
    # change name of column
    budget.columns = ['budget']
    # get all features with values greater than 1
    excluded = budget[budget[ 'budget' ] > 1].index
    # get all other names
    included = budget[budget[ 'budget' ] <= 1].index
    
    return excluded, included

def best_threshold(fpr, tpr, thresholds):
    """given the roc curve information, it returns the best threshold accoridng to the gmean

    Args:
        fpr (np.array): false positiva rate values
        tpr (np.array): true positive rate values
        thresholds (np.array): thresholds found for the ROC curve

    Returns:
        float: optimal threshold
    """
    # Calculate the G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    # gmeanOpt = round(gmean[index], ndigits = 4)
    # fprOpt = round(fpr[index], ndigits = 4)
    # tprOpt = round(tpr[index], ndigits = 4)

    return thresholdOpt