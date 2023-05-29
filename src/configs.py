from abc import ABC, abstractmethod
import yaml

from typing import Any, Union



PRT_NAME = '[Config]'

class Configurator:
    FIRST_LOAD = True
    def __init__(self, yaml_fp: str, with_partition=False) -> None:
        if Configurator.FIRST_LOAD:
            print(f'{PRT_NAME} Loading configuration from "{yaml_fp}".')
            Configurator.FIRST_LOAD = False
        if with_partition:
            yaml_load = self.from_yaml_all(yaml_fp)
        else:
            yaml_load = self.from_yaml(yaml_fp)
        for key in yaml_load:
            setattr(self, key, yaml_load[key])
            # getattr(self, key).__set_name__(self, key)

    @staticmethod
    def from_yaml(load_path) -> Union[dict, Any]:
        with open(load_path, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml
    
    @staticmethod
    def from_yaml_all(load_path) -> Union[dict, Any]:
        parsed_yaml = {}
        with open(load_path, 'r') as stream:
            try:
                for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                    parsed_yaml.update(data)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml


class _Configuration(ABC):
    """Base class for configuration/specification classes."""
    def __init__(self, config: Configurator) -> None:
        self._config = config
        self._load_config()

    @abstractmethod
    def _load_config(self):
        pass

    @classmethod
    def from_yaml(cls, yaml_fp: str, with_partition=False):
        config = Configurator(yaml_fp, with_partition)
        return cls(config)


class WarehouseSimConfiguration(_Configuration):
    """Configuration for warehouse simulation."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.scene = config.scene
        self.map_dir = config.map_dir
        self.map_file = config.map_file
        self.graph_file = config.graph_file
        self.mmp_cfg = config.mmp_cfg
        self.mpc_cfg = config.mpc_cfg
        self.dwa_cfg = config.dwa_cfg

        self.sim_width = config.sim_width
        self.sim_height = config.sim_height

        self.scale2nn = config.scale2nn
        self.scale2real = config.scale2real

        self.image_axis = config.image_axis
        self.corner_coords = config.corner_coords


class CircularRobotSpecification(_Configuration):
    """Specification class for circular robots."""
    def __init__(self, config: Configurator):
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts     # sampling time

        self.vehicle_width = config.vehicle_width
        self.vehicle_margin = config.vehicle_margin
        self.social_margin = config.social_margin
        self.lin_vel_min = config.lin_vel_min
        self.lin_vel_max = config.lin_vel_max
        self.lin_acc_min = config.lin_acc_min
        self.lin_acc_max = config.lin_acc_max
        self.ang_vel_max = config.ang_vel_max
        self.ang_acc_max = config.ang_acc_max


class WtaNetConfiguration(_Configuration):
    """Configuration class for WTA Net."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.device = config.device
        self.dim_out = config.dim_out
        self.dynamic_env = config.dynamic_env
        self.fc_input = config.fc_input
        self.input_channel = config.input_channel
        self.num_hypos = config.num_hypos
        self.obsv_len = config.obsv_len
        self.pred_len = config.pred_len
        
        self.batch_size = config.batch_size
        self.checkpoint_dir = config.checkpoint_dir
        self.early_stopping = config.early_stopping
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.weight_regularization = config.weight_regularization
        
        self.cell_width = config.cell_width
        self.x_max_px = config.x_max_px
        self.y_max_px = config.y_max_px
        
        self.data_name = config.data_name
        self.data_path = config.data_path
        self.label_csv = config.label_csv
        self.label_path = config.label_path
        self.model_path = config.model_path


class MpcConfiguration(_Configuration):
    """Configuration class for MPC Trajectory Generation Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts        # sampling time

        self.N_hor = config.N_hor  # control/pred horizon
        self.action_steps = config.action_steps # number of action steps (normally 1)

        self.ns = config.ns        # number of states
        self.nu = config.nu        # number of inputs
        self.nq = config.nq        # number of penalties
        self.Nother = config.Nother   # number of other robots
        self.nstcobs = config.nstcobs # dimension of a static-obstacle description
        self.Nstcobs = config.Nstcobs # number of static obstacles
        self.ndynobs = config.ndynobs # dimension of a dynamic-obstacle description
        self.Ndynobs = config.Ndynobs # number of dynamic obstacles

        self.max_solver_time = config.max_solver_time   # maximum time for the solver to run
        self.build_directory = config.build_directory   # directory to store the generated solver
        self.build_type = config.build_type             # type of the generated solver
        self.bad_exit_codes = config.bad_exit_codes     # bad exit codes of the solver
        self.optimizer_name = config.optimizer_name     # name of the generated solver

        self.lin_vel_penalty = config.lin_vel_penalty   # Cost for linear velocity control action
        self.lin_acc_penalty = config.lin_acc_penalty   # Cost for linear acceleration
        self.ang_vel_penalty = config.ang_vel_penalty   # Cost for angular velocity control action
        self.ang_acc_penalty = config.ang_acc_penalty   # Cost for angular acceleration
        self.qrpd = config.qrpd                         # Cost for reference path deviation
        self.qpos = config.qpos                         # Cost for position deviation each time step to the reference
        self.qvel = config.qvel                         # Cost for speed    deviation each time step to the reference
        self.qtheta = config.qtheta                     # Cost for heading  deviation each time step to the reference
        self.qpN = config.qpN                           # Terminal cost; error relative to final reference position       
        self.qthetaN = config.qthetaN                   # Terminal cost; error relative to final reference heading


class DwaConfiguration(_Configuration):
    """Configuration class for DWA Trajectory Generation Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts        # sampling time

        self.N_hor = config.N_hor  # control/pred horizon
        self.ns = config.ns        # number of states
        self.nu = config.nu        # number of inputs
        self.vel_resolution = config.vel_resolution
        self.ang_resolution = config.ang_resolution
        self.stuck_threshold = config.stuck_threshold
        self.q_goal_dir = config.q_goal_dir
        self.q_ref_deviation = config.q_ref_deviation
        self.q_speed = config.q_speed
        self.q_stc_obstacle = config.q_stc_obstacle
        self.q_dyn_obstacle = config.q_dyn_obstacle
        self.q_social = config.q_social