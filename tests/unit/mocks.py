import pytest
import pytest_mock as pm

from Quick.constants.random import SEED

# from fastai.tabular.all import (
#     # RandomSplitter,
#     # range_of
# )

def print_mock(mocker):

    # we will mock print expecting it to be called with 'hello, world'
    mocker.patch(
        'builtins.print',
        return_value='hello, world; successfully mocked',
        expected_input='hello, world_1'
    )

    print('hello, world') 

def mock_Tabular_Pandas(mocker, df):
    print('hello, world from mock_Tabular_Pandas')
    mocker.patch(
        'Quick.runners.utils.TabularPandas',
        return_value=df
    )


def mock_index_splitter(mocker):


    def t():
        print('successfully mocked index splitter')
        return 5

    mocker.patch(
        'Quick.runners.utils.IndexSplitter',
        return_value=t
    )

    print('hello, world')