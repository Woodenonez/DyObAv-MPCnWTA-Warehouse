from typing import Union, Tuple, List
from util.basic_datatype import *
from util.basic_objclass import *

class GloablPathPlanner:
    def __init__(self, external_path:Path=None) -> None:
        self.__scheduled_path(external_path)

    def __scheduled_path(self, external_path:Path=None):
        self.__next_node_idx = 0
        self.global_path = external_path
        self.start_node = None # needs to be set according to the robot's location
        if self.global_path:
            self.next_node  = self.global_path[self.__next_node_idx]
            self.final_node = self.global_path[-1]
        else:
            self.next_node = None
            self.final_node = None
            
    def set_start_node(self, start:Node):
        self.global_path.insert(0, start)
        self.start_node = start

    def move_to_next_node(self):
        self.__next_node_idx += 1
        try:
            self.next_node = self.global_path[self.__next_node_idx]
        except:
            self.__next_node_idx -= 1
