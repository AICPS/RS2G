import os
import sys

#import check_gpu as cg
#os.environ['CUDA_VISIBLE_DEVICES'] = cg.get_free_gpu()
sys.path.append(os.path.dirname(sys.path[0]))
from learning.util.ss_scenegraph_trainer import SS_Scenegraph_Trainer
from util.config_parser import configuration
import wandb

#python 3_train_model.py --yaml_path C:\users\harsi\research\roadscene2vec\roadscene2vec\config\learning_config.yaml  

def train_Trainer(learning_config):
    ''' Training the dynamic kg algorithm with different attention layer choice.'''

    #wandb setup 
    wandb_arg = wandb.init(config=learning_config, 
                        project=learning_config.wandb_config['project'], 
                        entity=learning_config.wandb_config['entity'])
    if learning_config.model_config['model_save_path'] == None:
        learning_config.model_config['model_save_path'] = "/media/data1/arnav/saved_graph_models/" + wandb_arg.name + ".pt" # save models with wandb nametag instead of manual naming.

    assert learning_config.model_config['model'] in ['ssmrgcn', 'ssvmrgcn'], 'This script only supports the SSMRGCN and SSVMRGCN.'

    if learning_config.training_config["dataset_type"] == "scenegraph":
        trainer = SS_Scenegraph_Trainer(learning_config, wandb_arg)
        trainer.split_dataset()
        trainer.build_model()
        trainer.learn()
        trainer.evaluate()
    else:
        raise ValueError("Task unrecognized")

    trainer.save_model()

if __name__ == "__main__":
    # the entry of dynkg pipeline training
    learning_config = configuration(sys.argv[1:])
    train_Trainer(learning_config)