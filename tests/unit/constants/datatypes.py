import numpy as np

from typing import (
    Any,
    Dict,
    List
)

SAMPLE_BASE_TYPE_DATA: Dict[str, Any] = {
    'name': 'test',
    'X_train': np.array([1, 2, 3]),
    'y_train': np.array([1, 2, 3]),
    'classes': ['a', 'b', 'c'],
    'target_label': 'test'
}
