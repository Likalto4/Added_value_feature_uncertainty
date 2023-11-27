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
import collections
from tqdm import tqdm
from utils import dataset_INCan, patient

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
    dataset = dataset_INCan()
    param_path = repo_path / 'data/param_files/Param_2D_sym.json' #path of parameter file
    extractor = extractor_settings(param_path, show=False)
    features_dir = repo_path / 'data/features/pure'

    logger = radiomics.logging.getLogger('radiomics')
    logger.setLevel(radiomics.logging.ERROR)

    for rad in ['L','M', 'V']:
        for time in ['1', '2']:

            df_all = pd.DataFrame()

            count_bar = tqdm(dataset.pat_num, total=dataset.len, desc=f'{rad}{time}')
            for id_num in count_bar:
                # read data
                pat = patient(pat_num=id_num)
                im_sitk = pat.get_im(sequence='SET', format='sitk')
                seg_sitk = pat.get_seg(rad=rad, time=time, format='sitk')

                #extract
                result = extractor.execute(im_sitk, seg_sitk) # Extract features

                # get features and store in dataframe
                df = features_df(result, id_num)

                # stack to df_all
                df_all = pd.concat([df_all, df])

            count_bar.close()

            # find sum_average_column and remove
            sum_avg_col = df_all.columns[df_all.columns.str.contains('SumAverage')]
            df_all = df_all.drop(columns=sum_avg_col)

            # save features
            df_all.to_csv(features_dir / f'features_{rad}_{time}.csv')
    
    # create mean feature vector
    # get all csv files in the directory
    all_df = None
    for rad in ['L', 'M', 'V']:
        for time in ['1', '2']:
            ind_path = features_dir / f'features_{rad}_{time}.csv'
            # if the file does not exist, skip
            if not ind_path.exists():
                print(f'{ind_path} does not exist')
                continue
            df = pd.read_csv(ind_path)
            # add col;umn with the string of the file name
            df['file'] = rad + time
            # concat to the main df
            all_df = pd.concat([all_df, df], axis=0) if all_df is not None else df

    # all_df averaged by pat_num
    feat_vector = all_df.groupby(by='pat_num', axis=0).mean(numeric_only=True)
    # save as  csv
    feat_vector.to_csv(repo_path / 'data/features/feat_vector.csv')

if __name__ == '__main__':
    main()