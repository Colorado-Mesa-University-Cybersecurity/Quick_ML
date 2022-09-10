'''

'''

import contextlib, pathlib

import fastai
import pandas as pd
import sklearn

from collections import ChainMap

from sklearn.neighbors import KNeighborsClassifier

from fastai.tabular.all import (
    Categorify,
    FillMissing, 
    Normalize,
    RandomSplitter,
    TabularPandas,
    range_of,
    tabular_learner
)

from sklearn.metrics import (
    accuracy_score, 
    classification_report
) 

from .utils import (
    create_feature_sets
)

from ..constants.random import SEED

from ..datatypes.model import (
    Model_data,
    ModelData
)


def run_sk_experiment(
    df: pd.DataFrame, 
    target_label: str, 
    split: float = 0.2,
    batch_size: int = 64, 
    categorical : list = ['Protocol'], 
    name: str or None = None,
    leave_out: list = [], 
    model = KNeighborsClassifier()
) -> Model_data or ModelData:
    '''
        Run binary classification using K-Nearest Neighbors
        returns the 10-tuple Model_data
    '''

    # First we split the features into the dependent variable and 
    # continous and categorical features
    print(df.shape)

    if name is None:
        name = f'SKLearn Classifier: {model.__name__}'
 
    categorical_features, continuous_features = create_feature_sets(
        df, 
        target_label, 
        leave_out = leave_out, 
        categorical = categorical
    )

    # Next, we set up the feature engineering pipeline, namely filling missing values
    # encoding categorical features, and normalizing the continuous features
    # all within a pipeline to prevent the normalization from leaking details
    # about the test sets through the normalized mapping of the training sets
    procs = [FillMissing, Categorify, Normalize]
    splits = RandomSplitter(valid_pct=split, seed=SEED)(range_of(df))
    
    
    # The dataframe is loaded into a fastai datastructure now that 
    # the feature engineering pipeline has been set up
    to = TabularPandas(
        df           , y_names=target_label          , 
        splits=splits, cat_names=categorical_features,
        procs=procs  , cont_names=continuous_features, 
    )


    # The dataframe is then converted into a fastai dataset
    try:
        dls = to.dataloaders(bs=batch_size)
        print('hello')
    except:
        dls = to
    
    temp_model = tabular_learner(dls)
    classes : list = list(temp_model.dls.vocab)

    # extract the name from the path
    p = pathlib.Path(name)
    name: str = str(p.parts[-1])


    # We extract the training and test datasets from the dataframe
    X_train = to.train.xs.reset_index(drop=True)
    X_test = to.valid.xs.reset_index(drop=True)
    y_train = to.train.ys.values.ravel()
    y_test = to.valid.ys.values.ravel()


    # Now that we have the train and test datasets, we set up a gridsearch of the K-NN classifier
    # using SciKitLearn and print the results 
    # params = {"n_neighbors": range(1, 50)}
    # model = GridSearchCV(KNeighborsClassifier(), params)

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    report = classification_report(y_test, prediction)
    print(report)
    print(f'\tAccuracy: {accuracy_score(y_test, prediction)}\n')

    # print("Best Parameters found by gridsearch:")
    # print(model.best_params_)


   # we add a target_type_ attribute to our model so yellowbrick knows how to make the visualizations
    if len(classes) == 2:
        model.target_type_ = 'binary'
    elif len(classes) > 2:  
        model.target_type_ = 'multiclass'
    else:
        print('Must be more than one class to perform classification')
        raise ValueError('Wrong number of classes')

    model_data: Model_data = Model_data(name, model, classes, X_train, y_train, X_test, y_test, to, dls, f'K_Nearest_Neighbors')

    # Now that the classifier has been created and trained, we pass out our training values
    # for analysis and further experimentation
    return model_data




def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
        'fastai': f'\t\t{fastai.__version__}',
        'pandas': f'\t\t{pd.__version__}',
        'sklearn': f'\t\t{sklearn.__version__}',
    })

    return versions

