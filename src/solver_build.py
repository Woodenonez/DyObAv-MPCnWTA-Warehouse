import os
import pathlib

from configs import MpcConfiguration, CircularRobotSpecification

from basic_motion_model import motion_model
from pkg_mpc_tracker.solver_build import mpc_builder

def return_cfg_path(fname: str) -> str:
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = os.path.join(root_dir, "config", fname)
    return cfg_path

def load_mpc_config(fname: str) -> MpcConfiguration:
    """Load the MPC configuration."""
    return MpcConfiguration.from_yaml(return_cfg_path(fname))

def load_robot_spec(fname: str) -> CircularRobotSpecification:
    """Load the robot specification."""
    return CircularRobotSpecification.from_yaml(return_cfg_path(fname))

if __name__ == "__main__":
    cfg_fname = "mpc_fast.yaml"
    config_mpc = load_mpc_config(cfg_fname)
    config_robot = load_robot_spec(cfg_fname)
    mpc_module = mpc_builder.MpcModule(config_mpc, config_robot)
    mpc_module.build(motion_model.unicycle_model)