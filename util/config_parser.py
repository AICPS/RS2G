#Copied from https://github.com/AICPS/roadscene2vec
#Copyright (c) 2021 UC Irvine Advanced Integrated Cyber-Physical Systems Lab (AICPS)

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])), "config"))
from argparse import ArgumentParser
import yaml
from pathlib import Path

class configuration:
    def __init__(self, args, from_function = False):
        if type(args) != list:
            from_function = True
        if not(from_function):
            ap = ArgumentParser(description='The parameters for use-case 2.')
            ap.add_argument('--yaml_path', type=str, default="./IP-NetList.yaml", help="The path of yaml config file.")
            ap.add_argument('--model', type=str, default = None, help="model override.")
            ap.add_argument('--input_path', type=str, default = None, help="input_path override.")
            ap.add_argument('--task_type', type=str, default = None, help="task type override.")
            args_parsed = ap.parse_args(args)
            for arg_name in vars(args_parsed):
                self.__dict__[arg_name] = getattr(args_parsed, arg_name)
                self.yaml_path = Path(self.yaml_path).resolve()
            #handle command line overrides.
            if self.model != None:
                self.model_config['model'] = self.model
            if self.input_path != None:
                self.location_data['input_path'] = self.input_path
            if self.task_type != None:
                self.training_config['task_type'] = self.task_type
            
            # if self.model_config['graph_extraction'] == 'rule_based':
            #     self.training_config['load_lane_info'] = False
                    
        
        if from_function:
            self.yaml_path = Path(args).resolve()
        with open(self.yaml_path, 'r') as f:
            args = yaml.safe_load(f)
            for arg_name, arg_value in args.items():
                self.__dict__[arg_name] = arg_value


        
    @staticmethod
    def parse_args(yaml_path):
        return configuration(yaml_path,True)




if __name__ == "__main__":
    configuration(sys.argv[1:])