
import numpy as np

from collections import (
    ChainMap,
    namedtuple
)

from typing import (
    Any,
    List,
    NamedTuple,
    Union
)

from ..constants.datatypes import BASE_TYPE_SCHEMA

Base_Data = namedtuple('base_data', BASE_TYPE_SCHEMA)

class BaseData(NamedTuple):
    name: str
    X_train: np.ndarray