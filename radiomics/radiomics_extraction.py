# add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None


import radiomics
from radiomics import featureextractor
import pandas as pd
import SimpleITK as sitk
import collections
from tqdm import tqdm

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

def main():
    # settings
    valid_patients = pd.read_csv(repo_path / 'data/valid_patients.csv', header=0, dtype=str)
    image_dir = repo_path / 'data/images/SET'
    features_dir = repo_path / 'data/features/pure'
    features_dir.mkdir(parents=True, exist_ok=True)

    param_path = repo_path / 'data/param_files/Param_64bin_all_radiomics.json' #path of parameter file
    extractor = extractor_settings(param_path, show=False)

    logger = radiomics.logging.getLogger('radiomics')
    logger.setLevel(radiomics.logging.ERROR)

    for rad in ['L','M', 'V']:
        for time in ['1', '2']:

            df_all = pd.DataFrame()

            count_bar = tqdm(valid_patients.iterrows(), total=valid_patients.shape[0], desc=f'Extracting {rad}_{time}')
            for id_num in valid_patients['pat_num']:
                # read image
                im_path = image_dir / f'Pat_{id_num}_SET_SMC_to_CMC_1min.tif'
                im_sitk = sitk.ReadImage(str(im_path))
                # read segmentation
                seg_path = repo_path / f'data/fixed_binary_masks/{rad}_{time}_seg/{id_num}_G_{rad}.seg.nrrd' 
                seg_sitk = sitk.ReadImage(str(seg_path))

                #extract
                result = extractor.execute(im_sitk, seg_sitk) # Extract features

                # get features and store in dataframe
                df = features_df(result, id_num)

                # stack to df_all
                df_all = pd.concat([df_all, df])
                count_bar.update(1)

            count_bar.close()

            # find sum_average_column and remove
            sum_avg_col = df_all.columns[df_all.columns.str.contains('SumAverage')]
            df_all = df_all.drop(columns=sum_avg_col)

            # save features
            df_all.to_csv(features_dir / f'features_{rad}_{time}.csv')

if __name__ == '__main__':
    main()