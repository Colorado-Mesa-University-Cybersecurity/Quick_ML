'''
Here we write tests that simply check to see if code is reachable from Quick/__init__.py
'''
from typing import Callable, List
import tunnel
import Quick

from Quick.analysis import *
from Quick.cleaning import *
from Quick.core import *
from Quick.encoding import *
from Quick.models import *
from Quick.runners import *
from Quick.topology import *
from Quick.visualization import *


def connection_tests():
    print("Executing basic connection tests.")
    if run_tests([
        test_connection_analysis,
        test_connection_cleaning,
        test_connection_core,
        test_connection_encoding,
        test_connection_models,
        test_connection_runners,
        test_connection_topology,
        test_connection_visualization
    ]):
        print("Connection tests passed.")

def run_tests(tests: List[Callable[[], bool]]) -> bool:
    
    passing = True

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Test {test.__name__} failed.")
            print(e)
            passing = False

    return passing

if(__name__ == "__main__"):
    connection_tests()