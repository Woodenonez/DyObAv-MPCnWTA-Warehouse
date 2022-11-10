# This is will be a long list of nearly empty classes
# These classes are used to hint commonly-used data type.
from typing import List, Union, TypeVar, NewType
import casadi as cs
import numpy as np

#%% Common data type
class FilePath(str): pass
class FolderDir(str): pass

class SamplingTime(float): pass

class State(tuple): pass
class Action(tuple): pass
class NumpyState(np.ndarray): pass
class NumpyAction(np.ndarray): pass

class NodeIdx(int): pass

#%% CasADi data type
class CasadiState(cs.SX): pass
class CasadiAction(cs.SX): pass

#%% OpEN data type
class OpENSolution(object): # placeholder, just for reference
    def __init__(self) -> None:
        self.cost = 0.0
        self.exit_status = 'SomeStatus'
        self.solve_time_ms = 0.0
        self.solution = [0.0, 0.0]
