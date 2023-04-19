# RoadScene2Graph (RS2G)

## Description
This repository contains the code for our paper titled ["RS2G: Data-Driven Scene-Graph Extraction and Embedding for Robust Autonomous Perception and Scenario Understanding"](https://arxiv.org/abs/2304.08600)

The repository is based on the structure and code from roadscene2vec, available [here](https://github.com/AICPS/roadscene2vec).

The main changes are the addition of the RS2G [model](https://github.com/AICPS/RS2G/blob/main/learning/model/rs2g.py), the [RS2G_Trainer](https://github.com/AICPS/RS2G/blob/main/learning/util/rs2g_trainer.py), and the [RS2G execution script](https://github.com/AICPS/RS2G/blob/main/scripts/6_train_rs2g_model.py).

Configuration parameters for the RS2G model as well as the baselines are available in config/. All hyperparameter tuning can be done via changes to these config files.

Please cite our paper if you find our code or paper useful for your research:
```
@article{malawade2023rs2g,
      title={RS2G: Data-Driven Scene-Graph Extraction and Embedding for Robust Autonomous Perception and Scenario Understanding}, 
      author={Arnav Vaibhav Malawade and Shih-Yuan Yu and Junyao Wang and Mohammad Abdullah Al Faruque},
      year={2023},
      journal={arXiv preprint arXiv:2304.08600}
}
```

## Installation
Installation follows the same procedure as roadscene2vec. Please follow the installation instructions for that library located [here](https://github.com/AICPS/roadscene2vec/blob/main/README.md#general-python-setup). At a minimum, you will need Python 3, PyTorch, and PyTorch Geometric. 

## Datasets
Our synthetic datasets can be downloaded at the following link: https://drive.google.com/drive/folders/1Zpzfvt_4jlgEiI8eE0HdICu4xkoug5jq?usp=sharing.
We do not have permission to publicly share the HDD dataset or the Traffic-Anomaly dataset. If you would like to use these datasets or others, please follow the instructions from roadscene2vec for [real-image scene-graph extraction](https://github.com/AICPS/roadscene2vec#use-case-1-converting-an-ego-centric-observation-image-into-a-scene-graph).


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
