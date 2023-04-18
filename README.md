# RoadScene2Graph (RS2G)

## Description
This repository contains the code for our paper titled "RS2G: Data-Driven Scene-Graph Extraction and Embedding for Robust Autonomous Perception and Scenario Understanding"

The repository is based on the structure and code from roadscene2vec, available [here](https://github.com/AICPS/roadscene2vec).

The main changes are the addition of the RS2G [model](https://github.com/AICPS/RS2G/blob/main/learning/model/rs2g.py), the [RS2G_Trainer](https://github.com/AICPS/RS2G/blob/main/learning/util/rs2g_trainer.py), and the [RS2G execution script](https://github.com/AICPS/RS2G/blob/main/scripts/6_train_rs2g_model.py).

Configuration parameters for the RS2G model as well as the baselines are available in config/. All hyperparameter tuning can be done via changes to these config files.

# Installation
Installation follows the same procedure as roadscene2vec. Please follow the installation instructions for that library located [here](https://github.com/AICPS/roadscene2vec/blob/main/README.md#general-python-setup). At a minimum, you will need Python 3, PyTorch, and PyTorch Geometric. 


## Model Training and Evaluation

The execution script can be run as follows for each model:
```
#Rule-Based MRGCN baseline
python 3_train_model.py --yaml_path ../config/rule_graph_risk_config.yaml  

#CNN+LSTM baseline
python 3_train_model.py --yaml_path ../config/image_learning_config.yaml

#RS2G
python 6_train_rs2g_model.py --yaml_path ../config/rs2g_graph_risk_config.yaml
```

## Transfer Learning

To run transfer learning experiments, the following commands can be used:
```
#Rule-Based MRGCN baseline
python 7_transfer_model.py --yaml_path ../config/transfer_rule_graph_risk_config.yaml

#CNN+LSTM baseline
python 7_transfer_model.py --yaml_path ../config/transfer_image_learning_config.yaml

#RS2G
python 7_transfer_model.py --yaml_path ../config/transfer_ss_graph_risk_config.yaml
```
