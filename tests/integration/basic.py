'''
Here we write tests that simply check to see if code is reachable from Quick/__init__.py
'''
import sys
  
# setting path
sys.path.append('../../.')



import Quick

Quick.hello()