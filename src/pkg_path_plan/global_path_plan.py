import pandas as pd
import networkx as nx
from .path_plan_graph import dijkstra

from ._path import PathNode, PathNodeList

from typing import Union, Tuple, List, Any

class GlobalPathPlanner:
    """Plan the global paths according to detailed or rough schedules.
    The detailed schedule includes full path for each robot, while the rough schedule only includes the start and end nodes.

    1. Detailed schedule:
        The schedule should include the specific time plan for each robot. A time plan is a list of nodes with corresponding ETAs.
        The necessary elements are: `robot_id`, `node_id`, `ETA` (estimated time of arrival).
    2. Rough schedule:
        The plan should include the start and end node of each robot with EDT.
        The necessary elements are: `robot_id`, `start_node`, `end_node`, `EDT` (estimated duration of travel).

    Attrs:
        `schedule`: <pd.DataFrame> The schedule of all robots.
        `robot_plan_dict`: <dict> The schedule of each robot.
        `robot_ids`: <list> The IDs of all robots.
        `G`: <nx.Graph> The graph of the environment.
    """
    def __init__(self) -> None:
        self.schedule = None
        self.robot_plan_dict:dict[Any, pd.DataFrame] = {}
        self._robot_ids = []
        self.G = None

    @property
    def robot_ids(self) -> list:
        return self._robot_ids

    def load_schedule_from_dataframe(self, new_schedule: pd.DataFrame):
        if 'ETA' in new_schedule.columns:
            self.with_detail = True
        elif 'EDT' in new_schedule.columns:
            self.with_detail = False
        else:
            raise ValueError("The schedule should include ETA or EDT.")
        
        if self.schedule is not None:
            self.schedule = pd.concat([self.schedule, new_schedule])
        else:   
            self.schedule = new_schedule
        self._robot_ids = self.schedule['robot_id'].unique()

        for robot_id in self._robot_ids:
            robot_plan:pd.DataFrame = self.schedule[self.schedule['robot_id'] == robot_id]
            robot_plan:pd.DataFrame = robot_plan.reset_index(drop=True)
            self.robot_plan_dict[robot_id] = robot_plan

    def load_schedule(self, file_path: str, csv_sep:str=',', header=0):
        new_schedule = pd.read_csv(file_path, sep=csv_sep, header=header)
        self.load_schedule_from_dataframe(new_schedule)

    def load_graph(self, G: nx.Graph):
        self.G = G

    def remove_schedule(self, robot_id: Any):
        self.schedule = self.schedule[self.schedule['robot_id'] != robot_id]
        self._robot_ids = self.schedule['robot_id'].unique()
        self.robot_plan_dict.pop(robot_id)

    def set_path(self, robot_id: Any, path_node_list: list, time_list:list=None):
        if time_list is None:
            time_list = [None for _ in range(len(path_node_list))]
        new_schedule = pd.DataFrame({'robot_id': robot_id, 'node_id': path_node_list, 'ETA': time_list})
        self.remove_schedule(robot_id)
        self.load_schedule_from_dataframe(new_schedule)
        
    def get_robot_schedule(self, robot_id: Any, time_offset: float) -> Tuple[list, list]:
        """
        Args:
            `robot_id`: The ID of the robot.
            `time_offset`: The time offset of the schedule. Normally it should be the current time.
            
        Returns:
            path_nodes (list): list of node ids
            path_times (list): list of time stamps
        """
        if self.with_detail:
            robot_schedule = self.robot_plan_dict[robot_id]
            path_nodes = robot_schedule['node_id'].tolist()
            path_times = robot_schedule['ETA'].tolist()
        else:
            if self.G is None:
                raise ValueError("The graph is not loaded.")
            source = self.robot_plan_dict[robot_id]['start_node']
            target = self.robot_plan_dict[robot_id]['end_node']
            edt = self.robot_plan_dict[robot_id]['EDT']
            path_nodes, section_length_list = self.get_shortest_path(self.G, source, target)
            path_times = [time_offset + x/sum(section_length_list)*edt for x in section_length_list]
        return path_nodes, path_times

    @staticmethod
    def get_shortest_path(graph: nx.Graph, source: Any, target: Any, algorithm:str='dijkstra'):
        """
        Args:
            `source`: The source node ID.
            `target`: The target node ID.
            `algorithm`: The algorithm used to find the shortest path. Currently only "dijkstra".
        Returns:
            `shortest_path`: The shortest path from source to target.
            `section_lengths`: The lengths of all sections in the shortest path.
        Notes:
            The weight key should be set to "weight" in the graph.
        """
        if algorithm == 'dijkstra':
            planner = dijkstra.DijkstraPathPlanner(graph)
            _, paths = planner.k_shortest_paths(source, target, k=1, get_coords=False)
            shortest_path = paths[0]
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")
        if len(shortest_path) > 2:
            section_lengths = [graph.edges[shortest_path[i], shortest_path[i+1]]['weight'] for i in range(len(shortest_path)-1)]
        return shortest_path, section_lengths
