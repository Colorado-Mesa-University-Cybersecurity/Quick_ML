import pytest
from tests.unit.mocks import print_mock
import tunnel
import sklearn

from sklearn.datasets import load_iris

from tests.unit.constants.t import *

def test_mocking(mocker):

    mocker.patch('test_examples.load_iris', return_value='hello, world; successfully mocked 1')

    print(load_iris())

def test_mocker(mocker):

    print_mock(mocker)

    mocker.patch('test_examples.load_iris', return_value=print)

    load_iris('hello')('world')

# def test_
