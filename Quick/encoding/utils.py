'''

'''

from collections import ChainMap

from ..datatypes.model import (
    Model_data, 
    ModelData
)

def calculate_correlations(model_data: Model_data or ModelData, target_label: str):
    '''
        Function merges together the encoded and standardized model data and labels to calculate pearson correlation
    '''

    encoded_data = model_data.X_train.copy()
    encoded_data[target_label] = model_data.y_train

    return encoded_data.corr()


def import_versions() -> ChainMap:
    '''
        Function will give a map of the versions of the imports used in this file
    '''
    versions: ChainMap = ChainMap({
    })

    return versions

