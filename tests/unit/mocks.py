import pytest
import pytest_mock as pm

def print_mock(mocker):

    print(mocker)
    print(pm.MockerFixture({}))
    # we will mock print expecting it to be called with 'hello, world'
    mocker.patch(
        'builtins.print',
        return_value='hello, world; successfully mocked',
        expected_input='hello, world_1'
    )

    print('hello, world')