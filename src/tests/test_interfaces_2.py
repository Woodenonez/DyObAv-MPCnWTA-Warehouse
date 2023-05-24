import unittest
### Import the module to be tested
from interfaces.map_interface import MapInterface
from interfaces.kfmp_interface import KfmpInterface
from interfaces.dwa_interface import DwaInterface
### Import dependencies of the module to be tested or test functions
from basic_map.map_geometric import GeometricMap
from basic_map.map_occupancy import OccupancyMap
from basic_map.graph_basic import NetGraph
### Import other modules
import numpy as np
import torch


class TestInterfaces(unittest.TestCase):
    def test_function_1(self): # Test MapInterface.
        # Arrange
        map_interface = MapInterface("warehouse_sim_original")

        input_data_1 = {"pgm_fname": "mymap.pgm",
                        "occupancy_threshold": 120,
                        "inversed_pixel": True
        }
        input_data_2 = {"json_fname": "mygraph.json"}
        
        # Act
        occ_map = map_interface.get_occ_map_from_pgm(**input_data_1)
        geo_map = map_interface.cvt_occ2geo(occ_map, 0.5)
        graph = map_interface.get_graph_from_json(**input_data_2)
        
        # Assert
        expected_output = None
        self.assertIsInstance(occ_map, OccupancyMap)
        self.assertIsInstance(geo_map, GeometricMap)
        self.assertIsInstance(graph, NetGraph)

    def test_function_2(self): # Test DwaInterface.
        # Arrange
        map_interface = MapInterface("warehouse_sim_original")
        occ_map = map_interface.get_occ_map_from_pgm("mymap.pgm", 120, inversed_pixel=True)
        geo_map = map_interface.cvt_occ2geo(occ_map, 0.5)

        mpc_interface = DwaInterface("dwa_test.yaml", np.array([0,0,0]), geo_map)
        mpc_interface.update_global_path([(0,0),(1,1),(2,2),(3,3)])

        input_data_1 = {"mode": "work",
                        "dyn_obstacle_list": None
        }
        
        # Act
        output_1 = mpc_interface.run_step(**input_data_1)
        
        # Assert
        expected_output = None
        self.assertEqual(len(output_1), 4)

    def test_function_3(self): # Test KfmpInterface.# Arrange
        kfmp_interface = KfmpInterface("mpc_test.yaml", state_space=None)

        input_data_1 = {"input_traj": [[0,0], [1,1], [2,2], [3,3]],
        }
        
        # Act
        output_1 = kfmp_interface.get_motion_prediction(**input_data_1)
        
        # Assert
        expected_output = None
        self.assertEqual(len(output_1), 2)


if __name__ == '__main__':
    unittest.main()
