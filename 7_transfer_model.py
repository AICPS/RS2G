import os
import sys
sys.path.append('../../')
from roadscene2vec.learning.util.image_trainer import Image_Trainer
from roadscene2vec.learning.util.scenegraph_trainer import Scenegraph_Trainer
from roadscene2vec.learning.util.ss_scenegraph_trainer import SS_Scenegraph_Trainer
from roadscene2vec.util.config_parser import configuration
import wandb

#python 7_transfer_model.py --yaml_path ../config/transfer_rule_graph_risk_config.yaml

def train_Trainer(learning_config):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''

    #wandb setup 
    wandb_arg = wandb.init(config=learning_config, 
                        project=learning_config.wandb_config['project'], 
                        entity=learning_config.wandb_config['entity'])
    if learning_config.model_config['model_save_path'] == None:
        learning_config.model_config['model_save_path'] = "/media/data1/arnav/saved_graph_models/" + wandb_arg.name + ".pt" # save models with wandb nametag instead of manual naming.

    if learning_config.training_config["dataset_type"] == "real":
        trainer = Image_Trainer(learning_config, wandb_arg)
        trainer.split_dataset()
        trainer.load_model()
        trainer.eval_model(current_epoch=0)
        
    elif learning_config.training_config["dataset_type"] == "scenegraph":
        if learning_config.model_config['model'] in ['ssmrgcn', 'ssvmrgcn']:
            trainer = SS_Scenegraph_Trainer(learning_config, wandb_arg)
        else:
            trainer = Scenegraph_Trainer(learning_config, wandb_arg)
        trainer.build_transfer_learning_dataset()
        trainer.load_model()
        trainer.evaluate_transfer_learning()
    else:
        raise ValueError("Task unrecognized")

    trainer.save_model()

if __name__ == "__main__":
    # the entry of dynkg pipeline training
    learning_config = configuration(sys.argv[1:])
    train_Trainer(learning_config)