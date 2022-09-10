'''

'''

import fastai
import pandas as pd

from collections import ChainMap

from fastai.data.all import DataLoaders

from fastai.tabular.all import (
    RandomSplitter,
    TabularPandas,
    range_of,
    tabular_learner
)

from ..constants.random import SEED


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


def create_dataloaders(
    df: pd.DataFrame,
    target_label: str,
    categorical_features: list,
    continuous_features: list,
    procs: list,
    batch_size: int,
    split: float
) -> tuple:
    '''
        Function will create the data loaders for the experiment
    '''

    splits = RandomSplitter(valid_pct = split, seed=SEED)(range_of(df))

    to = TabularPandas(
        df, 
        procs=procs, 
        cat_names=categorical_features, 
        cont_names=continuous_features, 
        y_names=target_label, 
        splits=splits
    )

    try:
        dls = to.dataloaders(bs=batch_size)
    except:
        dls = to

    dls.tabular_object = to

    return dls


def create_splits_from_tabular_object(to: TabularPandas) -> tuple:
    '''
        Function will create the splits from the tabular object
    '''

    # We extract the training and test datasets from the dataframe
    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()

    return X_train, X_test, y_train, y_test


def get_classes_from_dls(dls: DataLoaders) -> list:
    '''
        Function will return the classes from the dataloaders
    '''

    temp_model = tabular_learner(dls)

    return list(temp_model.dls.vocab)


def get_target_type(classes: list) -> str:
    '''
        Function will return the type of classification problem
    '''

    if len(classes) == 2:
        target_type_ = 'binary'
    elif len(classes) > 2:  
        target_type_ = 'multiclass'
    else:
        print('Must be more than one class to perform classification')
        raise ValueError('Wrong number of classes')

    return target_type_



def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'pandas': f'\t\t{pd.__version__}',
    })

    return versions

