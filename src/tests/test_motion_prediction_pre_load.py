import unittest
### Import the module to be tested
from pkg_motion_prediction import pre_load
from pkg_motion_prediction.data_handle.data_handler import DataHandler
from pkg_motion_prediction.data_handle.dataset import ImageStackDataset
from pkg_motion_prediction.network_manager import NetworkManager
### Import dependencies of the module to be tested or test functions
from configs import WtaNetConfiguration
### Import other modules
import tests.test_helpers as helpers


class TestMotionPredLoad(unittest.TestCase):
    def test_function_1(self): # Test pre_load.load_param/path().
        # Arrange
        input_data_1 = {"root_dir": helpers.ROOT_DIR,
                        "config_file": "wsd_1t20_train.yaml",
        }
        
        # Act
        output_1 = pre_load.load_config(**input_data_1)
        output_2 = pre_load.load_path(output_1)
        
        # Assert
        expected_output = None
        self.assertIsInstance(output_1, WtaNetConfiguration)
        self.assertEqual(len(output_2), 3)

    def test_function_2(self): # Test pre_load.load_data().
        # Arrange
        config = pre_load.load_config(root_dir=helpers.ROOT_DIR, config_file="wsd_1t20_train.yaml")
        paths = pre_load.load_path(config)

        input_data_1 = {"config": config,
                        "paths": paths,
                        "transform": None,
                        "load_for_test": True,
        }

        # Act
        output_1 = pre_load.load_data(**input_data_1)
        
        # Assert
        expected_output = None
        self.assertIsInstance(output_1[0], ImageStackDataset)
        self.assertIsInstance(output_1[1], DataHandler)

    def test_function_3(self): # Test pre_load.load_manager().
        # Arrange
        config = pre_load.load_config(root_dir=helpers.ROOT_DIR, config_file="wsd_1t20_train.yaml")

        input_data_1 = {"config": config,
                        "loss": {"meta": None, "base": None, "metric": None},
        }

        # Act
        output_1 = pre_load.load_manager(**input_data_1)
        
        # Assert
        expected_output = None
        self.assertIsInstance(output_1, NetworkManager)


if __name__ == '__main__':
    unittest.main()
