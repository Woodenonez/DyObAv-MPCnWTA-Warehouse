import unittest
### Import the module to be tested
from pkg_mpc_tracker import trajectory_tracker
### Import dependencies of the module to be tested or test functions
import casadi.casadi as cs
from casadi.tools import is_equal
### Import other modules
import tests.test_helpers as helpers


def almost_equal(x, y, tol=1e-3):
    return cs.fabs(x-y)<tol


class TestTrajectoryTracker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTrajectoryTracker, self).__init__(*args, **kwargs)
        cfg_mpc = helpers.load_mpc_config("mpc_test.yaml")
        cfg_robot = helpers.load_robot_spec("mpc_test.yaml")
        self.tracker = trajectory_tracker.TrajectoryTracker(cfg_mpc, cfg_robot, verbose=True)

    def test_function_1(self): # Test mpc_helper.dist_to_points_square().
        # Arrange
        input_data = {"point": cs.SX([[0,0]]), 
                      "points": cs.SX([[1,0],[2,0]])}
        
        # Act
        # output = self.tracker.dist_to_points_square(**input_data)
        
        # Assert
        # expected_output = cs.SX([[1,4]])
        # self.assertTrue(is_equal(output, expected_output))
        

if __name__ == '__main__':
    unittest.main()
