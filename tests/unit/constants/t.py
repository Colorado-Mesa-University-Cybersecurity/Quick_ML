import pytest

@pytest.mark.parametrize('n', [1, 2, 3])
def test_folder_resolution(n: int):
    print(f'test_folder_resolution: {n}')
    assert True

def test_folder_resolution_without_params():
    print('test_folder_resolution_without_params')
    assert True