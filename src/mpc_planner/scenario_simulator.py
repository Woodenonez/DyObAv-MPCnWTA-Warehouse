import math

class Simulator:
    def __init__(self, index, inflate_margin) -> None:
        if index is None:
            self.__hint()
            index = int(input('Please select a simulation index:'))
        self.idx = index
        self.inflate_margin = inflate_margin
        self.__intro()
        self.load_map_and_obstacles()

    def __hint(self):
        print('='*30)
        print('Index 0 - Test cases.')
        print('Index 1 - Single object, crosswalk.')
        print('Index 2 - Multiple objects, road crossing.')
        print('Index 3 - Single objects, crashing.')
        print('Index 4 - Single objects, following.')

    def __intro(self):
        assert(self.idx in [0,1,2,3,4]),(f'Index {self.idx} not found!')
        self.__hint()
        print(f'[{self.idx}] is selected.')
        print('='*30)

    def load_map_and_obstacles(self, test_graph_index=11):
        if self.idx == 0:
            from mpc_planner.test_maps.test_graphs import Graph
            from mpc_planner.obstacle_scanner.test_dynamic_obstacles import ObstacleScanner
            self.graph = Graph(inflate_margin=self.inflate_margin, index=test_graph_index)
            self.graph.processed_obstacle_list[3].pop(1) # XXX
            self.scanner = ObstacleScanner(self.graph)
            self.start = self.graph.start
            self.waypoints = [self.graph.end]
        elif self.idx == 1:
            from mpc_planner.test_maps.mmc_graph import Graph
            from mpc_planner.obstacle_scanner.mmc_dynamic_obstacles import ObstacleScanner
            self.start = (0.6, 3.5, math.radians(0))
            self.waypoints = [(15.4, 3.5, math.radians(0))]
            self.graph = Graph(inflate_margin=self.inflate_margin)
            self.scanner = ObstacleScanner()
        elif self.idx == 2:
            from mpc_planner.test_maps.mmc_graph2 import Graph
            from mpc_planner.obstacle_scanner.mmc_dynamic_obstacles2 import ObstacleScanner
            self.start = (7, 0.6, math.radians(90))
            self.waypoints = [(7, 11.5, math.radians(90)), (7, 15.4, math.radians(90))]
            self.graph = Graph(inflate_margin=self.inflate_margin)
            self.scanner = ObstacleScanner()
        elif self.idx == 3:
            from mpc_planner.test_maps.mmc_graph import Graph
            from mpc_planner.obstacle_scanner.crash_dynamic_obstacles import ObstacleScanner
            self.start = (0.6, 3.5, math.radians(0))
            self.waypoints = [(15.4, 3.5, math.radians(0))]
            self.graph = Graph(inflate_margin=self.inflate_margin, with_stc_obs=False)
            self.scanner = ObstacleScanner()
        elif self.idx == 4:
            from mpc_planner.test_maps.mmc_graph import Graph
            from mpc_planner.obstacle_scanner.follow_dynamic_obstacles import ObstacleScanner
            self.start = (0.6, 3.5, math.radians(0))
            self.waypoints = [(15.4, 3.5, math.radians(0))]
            self.graph = Graph(inflate_margin=self.inflate_margin, with_stc_obs=False)
            self.scanner = ObstacleScanner()
        else:
            raise ModuleNotFoundError
        
        return self.graph, self.scanner

        