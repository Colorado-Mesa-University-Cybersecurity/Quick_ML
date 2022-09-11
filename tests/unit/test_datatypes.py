import pytest
import numpy as np
import pprint

# from .tunnel import *
import tunnel

from tests.unit.constants.datatypes import (
    SAMPLE_BASE_TYPE_DATA
)

from tests.unit.utils.printer import (
    pretty
)

from Quick.datatypes.base import *

# @pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
# def test_base_data(n: int):
def test_base_data():
    data = SAMPLE_BASE_TYPE_DATA

    bd_1 = Base_Data(**data)
    bd_2 = BaseData(**data)
    bd_2.validate(verbose=True)
    


    for field in bd_2._fields:
        print(field, 
        bd_2.__getattribute__(field) )

        field_value = bd_2.__getattribute__(field)

        comparison = (field_value == data[field])
        if type(comparison) == np.ndarray:
            comparison = comparison.all()
        
        assert comparison

        
        comparison = (field_value == bd_1.__getattribute__(field))
        if type(comparison) == np.ndarray:
            comparison = comparison.all()
        
        assert comparison

    