'''
Here we write tests that simply check to see if code is reachable from Quick/__init__.py
'''

# this bit of code allows us to import Quick while being in a different directory that 
#    Quick is contained in
import sys
sys.path.append('../../.')
  



import Quick


if(__name__ == "__main__"):
    Quick.hello()