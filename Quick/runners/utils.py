'''

'''

import fastai
import pandas as pd

from collections import ChainMap


def create_feature_sets(
    df: pd.DataFrame, 
    target_label: str, 
    leave_out: list = [],
    categorical: list = [],
) -> tuple:
    '''
        Function will create the categorical and continuous feature sets for the dataframe
    '''

    categorical_features: list = []
    untouched_features  : list = []

    for x in leave_out:
        if x in df.columns:
            untouched_features.append(x)

    for x in categorical:
        if x in df.columns:
            categorical_features.append(x)

    continuous_features = list(
        set(df) - 
        set(categorical_features) - 
        set([target_label]) - 
        set(untouched_features)
    )

    return categorical_features, continuous_features



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
    })

    return versions

