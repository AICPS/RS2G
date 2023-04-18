#Copied from https://github.com/AICPS/roadscene2vec
#Copyright (c) 2021 UC Irvine Advanced Integrated Cyber-Physical Systems Lab (AICPS)

import os
import sys, pdb
sys.path.append(os.path.dirname(sys.path[0]))
from learning.util.image_trainer import Image_Trainer
from learning.util.scenegraph_trainer import Scenegraph_Trainer
from util.config_parser import configuration
import wandb

#Usage:
#python 3_train_model.py --yaml_path ../config/rule_graph_risk_config.yaml

def train_Trainer(learning_config):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''

    #wandb setup 
    wandb_arg = wandb.init(config=learning_config, 
                        project=learning_config.wandb_config['project'], 
                        entity=learning_config.wandb_config['entity'])
    if learning_config.model_config['model_save_path'] == None:
        learning_config.model_config['model_save_path'] = "./saved_graph_models/" + wandb_arg.name + ".pt" # save models with wandb nametag instead of manual naming.

    if learning_config.training_config["dataset_type"] == "real":
        trainer = Image_Trainer(learning_config, wandb_arg)
        trainer.split_dataset()
        trainer.build_model()
        trainer.learn()
        
    elif learning_config.training_config["dataset_type"] == "scenegraph":
        trainer = Scenegraph_Trainer(learning_config, wandb_arg)
        trainer.split_dataset()
        trainer.build_model()
        trainer.learn()
    else:
        raise ValueError("Task unrecognized")

    trainer.save_model()

if __name__ == "__main__":
    # the entry of dynkg pipeline training
    learning_config = configuration(sys.argv[1:])
    train_Trainer(learning_config)