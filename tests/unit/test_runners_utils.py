'''
    A test file for the runners module.
'''

import tunnel, pytest

from sklearn.model_selection import (
    StratifiedShuffleSplit
)

from tests.unit.utils.printer import (
    pretty
)

from tests.unit.constants.runner import (
    DLS_KEYS,
    SAMPLE_BATCHS,
    SAMPLE_CATEGORICAL_FEATURES,
    SAMPLE_CONTINUOUS_FEATURES,
    SAMPLE_DATAFRAME,
    SAMPLE_SPLITS,
    SAMPLE_TARGET_LABEL,
    SAMPLE_UNTOUCHED_FEATURES,
    TABULAR_OBJECT_KEYS,
)

from Quick.runners.utils import (
    create_feature_sets,
    create_dataloaders,
    create_cv_dataloaders
)

from Quick.constants.runners import (
    DEFAULT_PROCS
)

@pytest.mark.parametrize('n', [1, 2, 3])
def test_create_feature_sets(n: int):
    '''
        A test for the create_feature_sets function.
    '''

    categorical_features, continuous_features = create_feature_sets(
        SAMPLE_DATAFRAME,
        SAMPLE_TARGET_LABEL,
        leave_out = SAMPLE_UNTOUCHED_FEATURES[n],
        categorical = SAMPLE_CATEGORICAL_FEATURES[n]
    )

    for feature in categorical_features:
        assert feature in SAMPLE_CATEGORICAL_FEATURES[n]
    for feature in continuous_features:
        assert feature in SAMPLE_CONTINUOUS_FEATURES[n]
    for feature in categorical_features + continuous_features:
        assert feature not in SAMPLE_UNTOUCHED_FEATURES[n]


@pytest.mark.parametrize('n', [1, 2, 3])
def test_create_dataloaders(mocker, n: int):
    '''
        A test for the create_dataloaders function.
    '''
    
    dls = create_dataloaders(
        SAMPLE_DATAFRAME,
        SAMPLE_TARGET_LABEL,
        SAMPLE_CATEGORICAL_FEATURES[1],
        SAMPLE_CONTINUOUS_FEATURES[1],
        DEFAULT_PROCS,
        SAMPLE_BATCHS[n],
        SAMPLE_SPLITS[n],
    )

    assert len(dls) == 2


    assert dls.tabular_object is not None
    for field in dls.tabular_object.__dict__:
        assert field in TABULAR_OBJECT_KEYS
    
    for i in range(len(dls)):
        assert dls[i].__class__.__name__ == 'TabDataLoader'
        assert dls[i].bs == SAMPLE_BATCHS[n]
        
        for field in dls[i].__dict__:
            assert field in DLS_KEYS

@pytest.mark.parametrize('n', [1, 2, 3])
def test_create_cv_dataloaders(mocker, n: int):
    '''
        A test for the create_cv_dataloaders function.
    '''

    k_folds = 2
    SEED = 14
    df = SAMPLE_DATAFRAME
    target_label = SAMPLE_TARGET_LABEL

    ss = StratifiedShuffleSplit(n_splits=k_folds, random_state=SEED, test_size=1/k_folds)
    for _, (_, test_index) in enumerate(ss.split(df.copy().drop(target_label, axis=1), df[target_label])):

        dls = create_cv_dataloaders(
            SAMPLE_DATAFRAME,
            SAMPLE_TARGET_LABEL,
            SAMPLE_CATEGORICAL_FEATURES[1],
            SAMPLE_CONTINUOUS_FEATURES[1],
            DEFAULT_PROCS,
            SAMPLE_BATCHS[n],
            test_index
        )

        assert len(dls) == 2

        assert dls.tabular_object is not None
        for field in dls.tabular_object.__dict__:
            assert field in TABULAR_OBJECT_KEYS
        
        for i in range(len(dls)):
            assert dls[i].__class__.__name__ == 'TabDataLoader'
            assert dls[i].bs == SAMPLE_BATCHS[n]
            
            for field in dls[i].__dict__:
                assert field in DLS_KEYS