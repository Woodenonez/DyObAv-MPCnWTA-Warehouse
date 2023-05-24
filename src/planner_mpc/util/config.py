import yaml

""" 
    File that contains all the neccessary configuration parameters for the 
    MPC Trajectory Generation Module
"""

class Configurator:

    # class dotdict(dict):
    #     """dot.notation access to dictionary attributes"""
    #     __getattr__ = dict.get
    #     __setattr__ = dict.__setitem__
    #     __delattr__ = dict.__delitem__

    def __init__(self, yaml_fp):
        self.__prtname = '[CONFIG]'
        print(f'{self.__prtname} Loading configuration from "{yaml_fp}".')
        with open(yaml_fp, 'r') as stream:
            yaml_load = yaml.safe_load(stream)

        for key in yaml_load:
            setattr(self, key, yaml_load[key])

        # self.args = self.dotdict()
        # for key, value in yaml_load.items():
        #     self.args[key] = value
        print(f'{self.__prtname} Configuration done.')

