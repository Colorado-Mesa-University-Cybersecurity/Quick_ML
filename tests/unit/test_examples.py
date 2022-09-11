import pytest
from tests.unit.mocks import print_mock
import tunnel
import sklearn

from sklearn.datasets import load_iris

from Quick.constants.runners import DEFAULT_PROCS
from Quick.runners.sk import run_sk_experiment

def test_mocking(mocker):

    mocker.patch('test_examples.load_iris', return_value='hello, world; successfully mocked')

    print(load_iris())

def test_mocking_2(mocker):

    print_mock(mocker)

    print('hi')

