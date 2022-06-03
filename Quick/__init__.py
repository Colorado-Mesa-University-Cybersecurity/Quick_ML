# import analysis, cleaning, core, encoding, models, runners, topology, visualization

# import .analysis as analysis
from .analysis import *
from .cleaning import *
from .core import *
from .encoding import *
from .models import *
from .runners import *
from .topology import *
from .visualization import *


def hello():
    print("Hello")
    basic_test_analysis()
    basic_test_cleaning()
    basic_test_core()
    basic_test_encoding()
    basic_test_models()
    basic_test_runners()
    basic_test_topology()
    basic_test_visualization()

    