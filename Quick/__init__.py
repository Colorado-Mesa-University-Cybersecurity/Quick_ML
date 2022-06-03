'''
This is the main entrance point for the Quick library by Colorado Mesa University's Cyber Security Center
primary maintainer: James Halladay

Submodules:
    Analysis:
    Cleaning:
    Core:
    Encoding:
    Models:
    Runners:
    Topology:
    Visualization:


last updated: 2022-06-02
'''


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

    