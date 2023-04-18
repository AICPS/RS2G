import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
from learning.util.rs2g_trainer import RS2G_Trainer
from util.config_parser import configuration
import wandb

#Usage:
#python 3_train_model.py --yaml_path ../config/rs2g_graph_risk_config.yaml  

def train_Trainer(learning_config):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''

    #wandb setup 
    wandb_arg = wandb.init(config=learning_config, 
                        project=learning_config.wandb_config['project'], 
                        entity=learning_config.wandb_config['entity'])
    if learning_config.model_config['model_save_path'] == None:
        learning_config.model_config['model_save_path'] = "./saved_graph_models/" + wandb_arg.name + ".pt" # save models with wandb nametag instead of manual naming.

    assert learning_config.model_config['model'] in ['rs2g'], 'This script only supports the RS2G model.'

    if learning_config.training_config["dataset_type"] == "scenegraph":
        trainer = RS2G_Trainer(learning_config, wandb_arg)
        trainer.split_dataset()
        trainer.build_model()
        trainer.learn()
        trainer.evaluate()
    else:
        raise ValueError("Task unrecognized")

    trainer.save_model()

if __name__ == "__main__":
    learning_config = configuration(sys.argv[1:])
    train_Trainer(learning_config)