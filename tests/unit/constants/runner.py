import pandas as pd

from typing import (
    Callable,
    Dict,
    List
)



SAMPLE_TARGET_LABEL: str = 'target'

SAMPLE_DATAFRAME: pd.DataFrame = pd.DataFrame({
    'a': [1, 2, 3, 16, 17, 18, 1, 2, 3, 4, 5, 6],
    'b': [4, 5, 6, 16, 17, 18, 1, 2, 3, 4, 5, 6],
    'c': [7, 8, 9, 16, 17, 18, 1, 2, 3, 4, 5, 6],
    'd': [10, 11, 12, 16, 17, 18, 1, 2, 3, 4, 5, 6],
    'e': [13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6],
    SAMPLE_TARGET_LABEL: [16, 17, 18, 16, 17, 18, 16, 17, 18, 16, 17, 18],
})

SAMPLE_UNTOUCHED_FEATURES: Dict[int, List[str]] = {
    1: [],
    2: ['c'],
    3: [],
    4: ['a', 'b', 'c', 'd', 'e'],
}

SAMPLE_CATEGORICAL_FEATURES: Dict[int, List[str]] = {
    1: [],
    2: ['a'],
    3: ['a', 'b', 'c', 'd', 'e'],
    4: []
}

SAMPLE_CONTINUOUS_FEATURES: Dict[int, List[str]] = {
    1: ['a', 'b', 'c', 'd', 'e'],
    2: ['b', 'd', 'e'],
    3: [],
    4: []
}


SAMPLE_BATCHS: Dict[int, int] = {
    1: 64,
    2: 8,
    3: 2,
    4: 1024
}

SAMPLE_SPLITS: Dict[int, float] = {
    1: 0.2,
    2: 0.5,
    3: 0.1,
    4: 1
}

DLS_KEYS: List[str] = [
    'after_item', 'before_batch', 'after_batch', '__stored_args__', 
    'dataset'   , 'bs'          , 'shuffle'    , 'drop_last'      , 
    'indexed'   , 'n'           , 'pin_memory' , 'timeout'        , 
    'device'    , 'rng'         , 'num_workers', 'offs'           , 
    'fake_l'    , '_n_inp'      , '_types'     ,
]

TABULAR_OBJECT_KEYS: List[str] = [
    'dataloaders', 'items'     , 'y_names',  'device', 
    'cat_names'  , 'cont_names', 'procs'  , 'split'  ,
]