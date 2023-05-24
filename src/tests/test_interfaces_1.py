import unittest
### Import the module to be tested
from interfaces.mmp_interface import MmpInterface
from interfaces.map_interface import MapInterface
from interfaces.mpc_interface import MpcInterface
### Import dependencies of the module to be tested or test functions
from basic_map.map_geometric import GeometricMap
from basic_map.map_occupancy import OccupancyMap
from basic_map.graph_basic import NetGraph
### Import other modules
import numpy as np
import torch


class TestInterfaces(unittest.TestCase):
    def test_function_1(self): # Test MmpInterface.
        # Arrange
        mmp_interface = MmpInterface("wsd_1t20_train.yaml")
        config = mmp_interface.config

        input_data_1 = {"input_traj": [(0,0)],
                        "ref_image": torch.randn((config.y_max_px, config.x_max_px)),
                        "pred_offset": 5,
        }
        
        # Act
        output_1 = mmp_interface.get_motion_prediction(**input_data_1)
        
        # Assert
        expected_output = None
        self.assertEqual(len(output_1), 5)
        self.assertEqual(output_1[0].shape[0], config.num_hypos)

    def test_function_2(self): # Test MapInterface.
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

    def test_function_3(self): # Test MpcInterface.
        # Arrange
        map_interface = MapInterface("warehouse_sim_original")
        occ_map = map_interface.get_occ_map_from_pgm("mymap.pgm", 120, inversed_pixel=True)
        geo_map = map_interface.cvt_occ2geo(occ_map, 0.5)

        mpc_interface = MpcInterface("mpc_test.yaml", np.array([0,0,0]), geo_map)
        mpc_interface.update_global_path([(0,0),(1,1),(2,2),(3,3)])

        input_data_1 = {"mode": "work",
                        "full_dyn_obstacle_list": None,
                        "map_updated": True,
        }
        
        # Act
        output_1 = mpc_interface.run_step(**input_data_1)
        
        # Assert
        expected_output = None
        self.assertEqual(len(output_1), 5)


if __name__ == '__main__':
    unittest.main()
