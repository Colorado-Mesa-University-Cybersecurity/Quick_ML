
from fastai.callback.all import (
    ShowGraphCallback,
    minimum,
    slide,
    steep,
    valley
)

from fastai.tabular.all import (
    Categorify,
    FillMissing,
    Normalize
)


FIT: str = 'fit'
ONE_CYCLE: str = 'one_cycle'
FLAT_COS: str = 'flat_cos'

VALLEY: str = 'valley'
SLIDE: str = 'slide'
STEEP: str = 'steep'
MINIMUM: str = 'minimum'

LEARNING_RATE_OPTIONS: dict = {
    VALLEY: 0,
    SLIDE: 1, 
    STEEP: 2, 
    MINIMUM: 3
}

DEFAULT_CALLBACKS: list = [
    ShowGraphCallback
]

DEFAULT_PROCS: list = [
    FillMissing, 
    Categorify, 
    Normalize
]

DEFAULT_LR_FUNCS: list = [
    valley, 
    slide, 
    steep, 
    minimum
]