import unittest
### Import the module to be tested
from pkg_mpc_tracker.build import mpc_builder, mpc_cost, mpc_helper
### Import dependencies of the module to be tested or test functions
import casadi.casadi as cs
from casadi.tools import is_equal
### Import other modules
import tests.test_helpers as helpers


def almost_equal(x, y, tol=1e-3):
    return cs.fabs(x-y)<tol


class TestMpcHelper(unittest.TestCase):
    def test_function_1(self): # Test mpc_helper.dist_to_points_square().
        # Arrange
        input_data = {"point": cs.SX([[0,0]]), 
                      "points": cs.SX([[1,0],[2,0]])}
        
        # Act
        output = mpc_helper.dist_to_points_square(**input_data)
        
        # Assert
        expected_output = cs.SX([[1,4]])
        self.assertTrue(is_equal(output, expected_output))

    def test_function_2(self): # Test mpc_helper.dist_to_lineseg().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]), 
                        "line_segment": cs.SX([[3,2], [3,0]])}
        input_data_2 = {"point": cs.SX([[1,2]]),
                        "line_segment": cs.SX([[3,1], [3,0]])}
        
        # Act
        output_1 = mpc_helper.dist_to_lineseg(**input_data_1)
        output_2 = mpc_helper.dist_to_lineseg(**input_data_2)
        
        # Assert
        expected_output_1 = cs.SX([[2]])
        expected_output_2 = cs.SX([[cs.sqrt(5)]])
        self.assertTrue(is_equal(output_1, expected_output_1))
        self.assertTrue(is_equal(output_2, expected_output_2))

    def test_function_3(self): # Test mpc_helper.inside_ellipses().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "ellipse_param": [cs.SX([[1,1]]),
                                          cs.SX([[2,4]]),
                                          cs.SX([[1,1]]),
                                          cs.SX([[1,1]]),
                                          cs.SX([[0,0]])]}
        
        # Act
        output_1 = mpc_helper.inside_ellipses(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([[1, -3]])
        self.assertTrue(is_equal(output_1[0], expected_output_1[0]))
        self.assertTrue(almost_equal(output_1[1], expected_output_1[1]))

    def test_function_4(self): # Test mpc_helper.inside_cvx_polygon().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "b":  cs.SX([[ 0,2, 1,3]]),
                        "a0": cs.SX([[-1,1, 0,0]]),
                        "a1": cs.SX([[ 0,0,-1,1]]),
        }
        input_data_2 = {"point": cs.SX([[1,2]]),
                        "b":  cs.SX([[ 0,1, 0,1]]),
                        "a0": cs.SX([[-1,1, 0,0]]),
                        "a1": cs.SX([[ 0,0,-1,1]]),
        }
        
        # Act
        output_1 = mpc_helper.inside_cvx_polygon(**input_data_1)
        output_2 = mpc_helper.inside_cvx_polygon(**input_data_2)

        # Assert
        expected_output_1 = cs.SX([3])
        expected_output_2 = cs.SX([0])
        self.assertTrue(is_equal(output_1, expected_output_1))
        self.assertTrue(is_equal(output_2, expected_output_2))

    def test_function_5(self): # Test mpc_helper.outside_cvx_polygon().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "b":  cs.SX([[ 0,2, 1,3]]),
                        "a0": cs.SX([[-1,1, 0,0]]),
                        "a1": cs.SX([[ 0,0,-1,1]]),
        }
        input_data_2 = {"point": cs.SX([[1,2]]),
                        "b":  cs.SX([[ 0,1, 0,1]]),
                        "a0": cs.SX([[-1,1, 0,0]]),
                        "a1": cs.SX([[ 0,0,-1,1]]),
        }
        
        # Act
        output_1 = mpc_helper.outside_cvx_polygon(**input_data_1)
        output_2 = mpc_helper.outside_cvx_polygon(**input_data_2)

        # Assert
        expected_output_1 = cs.SX([0])
        expected_output_2 = cs.SX([1])
        self.assertTrue(is_equal(output_1, expected_output_1))
        self.assertTrue(is_equal(output_2, expected_output_2))

    def test_function_6(self): # Test mpc_helper.angle_between_vectors().
        # Arrange
        input_data_1 = {"l1": cs.SX([[0,0], [1,1]]), 
                        "l2": cs.SX([[0,1], [0,0]]),
                        "degrees": True}
        
        # Act
        output_1 = mpc_helper.angle_between_vectors(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([135])
        self.assertTrue(almost_equal(output_1, expected_output_1))


class TestMpcCost(unittest.TestCase):
    def test_function_1(self): # Test mpc_cost.cost_inside_cvx_polygon().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "b":  cs.SX([[ 0,2, 1,3]]),
                        "a0": cs.SX([[-1,1, 0,0]]),
                        "a1": cs.SX([[ 0,0,-1,1]]),
                        "weight": 2
        }
        
        # Act
        output_1 = mpc_cost.cost_inside_cvx_polygon(**input_data_1)
        
        # Assert
        expected_output = cs.SX([[18]])
        self.assertTrue(is_equal(output_1, expected_output))

    def test_function_2(self): # Test mpc_cost.cost_inside_ellipses().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "ellipse_param": [cs.SX([[1,1]]),
                                          cs.SX([[2,4]]),
                                          cs.SX([[1,1]]),
                                          cs.SX([[1,1]]),
                                          cs.SX([[0,0]])]}
        
        # Act
        output_1 = mpc_cost.cost_inside_ellipses(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([[1, 0]])
        self.assertTrue(is_equal(output_1[0], expected_output_1[0]))
        self.assertTrue(is_equal(output_1[1], expected_output_1[1]))

    def test_function_3(self): # Test mpc_cost.cost_control_actions().
        # Arrange
        input_data_1 = {"actions": cs.SX([[1,2,3]]),
                        "weights": cs.SX([[2,1,1]])}
        
        # Act
        output_1 = mpc_cost.cost_control_actions(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([[15]])
        self.assertTrue(is_equal(output_1, expected_output_1))

    def test_function_4(self): # Test mpc_cost.cost_control_jerks().
        # Arrange
        input_data_1 = {"actions": cs.SX([[1,2,3]]),
                        "last_actions": cs.SX([[0,1,1]]),
                        "weights": 2}
        
        # Act
        output_1 = mpc_cost.cost_control_jerks(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([[12]])
        self.assertTrue(is_equal(output_1, expected_output_1))

    def test_function_5(self): # Test mpc_cost.cost_fleet_collision().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "points": cs.SX([[0,0], [2,0]]),
                        "safe_distance": 2,
                        "weight": 2}
        input_data_2 = {"point": cs.SX([[1,2]]),
                        "points": cs.SX([[0,1], [2,0]]),
                        "safe_distance": 2,
                        "weight": 2}
        
        # Act
        output_1 = mpc_cost.cost_fleet_collision(**input_data_1)
        output_2 = mpc_cost.cost_fleet_collision(**input_data_2)

        # Assert
        expected_output_1 = cs.SX([[0]])
        expected_output_2 = cs.SX([[4]])
        self.assertTrue(is_equal(output_1, expected_output_1))
        self.assertTrue(is_equal(output_2, expected_output_2))

    def test_function_6(self): # Test mpc_cost.cost_refvalue_deviation().
        # Arrange
        input_data_1 = {"actual_value": cs.SX([1]),
                        "ref_value": cs.SX([0]),
                        "weight": 2}
        
        # Act
        output_1 = mpc_cost.cost_refvalue_deviation(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([2])
        self.assertTrue(is_equal(output_1, expected_output_1))

    def test_function_7(self): # Test mpc_cost.cost_refstate_deviation().
        # Arrange
        input_data_1 = {"state": cs.SX([[1,2]]),
                        "ref_state": cs.SX([[0,0]]),
                        "weights": 2}
        
        # Act
        output_1 = mpc_cost.cost_refstate_deviation(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([10])
        self.assertTrue(is_equal(output_1, expected_output_1))

    def test_function_8(self): # Test mpc_cost.cost_refpath_deviation().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "line_segments": cs.SX([[0,0],[1,0],[3,2]]),
                        "weight": 0.5}
        
        # Act
        output_1 = mpc_cost.cost_refpath_deviation(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([1.0])
        self.assertTrue(almost_equal(output_1, expected_output_1))

    def test_function_9(self): # Test mpc_cost.cost_refpoint_detach().
        # Arrange
        input_data_1 = {"point": cs.SX([[1,2]]),
                        "ref_point": cs.SX([[1,0]]),
                        "ref_distance": 1,
                        "weight": 2}
        
        # Act
        output_1 = mpc_cost.cost_refpoint_detach(**input_data_1)

        # Assert
        expected_output_1 = cs.SX([2])
        self.assertTrue(almost_equal(output_1, expected_output_1))

    
class TestMpcBuilder(unittest.TestCase):
    def test_function_1(self): # Test mpc_builder.MpcModule.build().
        # Arrange
        def unicycle_model(s, a, ts):
            d_s = ts * cs.vertcat(a[0]*cs.cos(s[2]), a[0]*cs.sin(s[2]), a[1])
            return s + d_s
        
        input_data_1 = {"motion_model": unicycle_model,
                        "use_tcp":  False,
                        "test": True,
        }
        
        config_mpc = helpers.load_mpc_config("mpc_test.yaml")
        config_robot = helpers.load_robot_spec("mpc_test.yaml")
        
        # Act
        mpc_module = mpc_builder.MpcModule(config_mpc, config_robot)
        output_1 = mpc_module.build(**input_data_1)
        
        # Assert
        expected_output = 1
        self.assertEqual(output_1, expected_output)


if __name__ == '__main__':
    unittest.main()
