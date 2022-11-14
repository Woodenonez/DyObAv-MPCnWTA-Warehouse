# This is a template of comments, you cannot run it.
# System import
import os
# External import
import numpy
# Key import
import gym
# Vis import
import matplotlib
# Custom import
from mymodule import myclass # If the class name has no ambiguity
from mymodule import myfunc # XXX This is bad
import mymodule # XXX Then later mymodule.myfunc. This is better

'''
File info:
    Author  - [XXX] # for important scripts, add your own name/nickname
    Date    - (from)[XXX] -> (to)[XXX]  # for important scripts
    Ref     - (literature)[XXX]
            - (website)[XXX]
    Exe     - (executable)[Yes]
File description:
    (What does this file do?)
File content:
    ClassA      <class> - (Basic usage).
    ClassB      <class> - (Basic usage).
    function_A  <func>  - (Basic usage).
    function_B  <func>  - (Basic usage).
Comments:
    (Things worthy of attention.)
'''

def function_A(x:int, y:int) -> int: # use IO hints
    '''
    Description:
        (What does this function do?)
    Arguments:
        :x <type> - (Description).
        :y <type> - (Description).
    Return:
        z <type> - (Description).
    Comments:
        (Things worthy of attention.)
    '''
    z = x+y
    return z

def function_B():
    pass

class ClassA():
    '''
    Description:
        (What does this class do?)
    Arguments:
        :x <type> - (Description).
        :y <type> - (Description).
    Attributes:
        :attr1 <type> - (Description).
        :attr2 <type> - (Description).
    Functions
        :func1 - (Description).
        :func2 - (Description).
    Comments:
        (Things worthy of attention.)
    '''
    def __init__(self, x:str, y:str):
        self.attr1 = x
        self.attr2 = y

    def func1(self):
        print('This is function 1.')

    def func2(self):
        print('This is function 2.')

class ClassB():
    pass

if __name__ == '__main__':
    do_something = True # this part for testing, if missing then 'Exe'=No