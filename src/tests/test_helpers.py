import os
import pathlib
from configs import MpcConfiguration, CircularRobotSpecification, WtaNetConfiguration

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

def return_cfg_path(fname: str) -> str:
    cfg_path = os.path.join(ROOT_DIR, "config", fname)
    return cfg_path

def load_mpc_config(fname: str) -> MpcConfiguration:
    """Load the MPC configuration."""
    return MpcConfiguration.from_yaml(return_cfg_path(fname))

def load_robot_spec(fname: str) -> CircularRobotSpecification:
    """Load the robot specification."""
    return CircularRobotSpecification.from_yaml(return_cfg_path(fname))

def load_wta_config(fname: str) -> WtaNetConfiguration:
    """Load the WTA configuration."""
    return WtaNetConfiguration.from_yaml(return_cfg_path(fname))