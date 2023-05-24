class ObstacleScanner(): # Template
    '''
    Description:
        Generate/read the information of dynamic obstacles, and form it into the correct format for MPC.
    Attributes:
        num_obstacles <int> - The number of (active) dynamic obstacles.
    Functions
        get_obstacle_info      <get> - Get the position, orientation, and shape information of a selected (idx) obstacle.
        get_full_obstacle_list <get> - Form the list for MPC problem.
    Comments:
        Other attributes and functions are for specific usage.
    '''
    def __init__(self):
        pass

    def get_obstacle_info(self, time:float, key:str):
        pass

    def get_full_obstacle_list(self, time:float, horizon:int):
        pass