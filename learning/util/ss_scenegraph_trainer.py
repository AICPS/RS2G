import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from learning.util.trainer import Trainer
from data.dataset import SceneGraphDataset
from torch_geometric.loader import DataListLoader
from learning.util.metrics import get_metrics, log_wandb, log_wandb_transfer_learning, get_graph_metrics


'''trainer for RS2G data-driven MRGCN model.'''

class SS_Scenegraph_Trainer(Trainer):
    def __init__(self, config, wandb_a = None):
        super(SS_Scenegraph_Trainer, self).__init__(config, wandb_a)
        self.scene_graph_dataset = SceneGraphDataset()
        self.feature_list = list()
        for i in range(self.config.model_config['num_of_classes']):
            self.feature_list.append("type_"+str(i))
        self.device = self.config.model_config['device']


    def split_dataset(self):
        if (self.config.training_config['task_type'] in ['sequence_classification','graph_classification','collision_prediction']):
            self.training_data, self.testing_data = self.build_scenegraph_dataset()
            self.total_train_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.training_data]) # used to compute frame-level class weighting
            self.total_test_labels  = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.testing_data])
            self.training_labels = [data['label'] for data in self.training_data]
            self.testing_labels  = [data['label'] for data in self.testing_data]
            if self.config.training_config['task_type'] == 'sequence_classification':
                self.class_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(self.training_labels), y=self.training_labels))
                if self.config.training_config["n_fold"] <= 1:
                    print("Number of Sequences Included: ", len(self.training_data))
                    print("Number of Testing Sequences Included: ", len(self.testing_data))
                    print("Num Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                    print("Number of Testing Labels in Each Class: " + str(np.unique(self.testing_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
            elif self.config.training_config['task_type'] == 'collision_prediction':
                self.class_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(self.total_train_labels), y=self.total_train_labels))
                if self.config.training_config["n_fold"] <= 1:
                    print("Number of Training Sequences Included: ", len(self.training_data))
                    print("Number of Testing Sequences Included: ", len(self.testing_data))
                    print("Number of Training Labels in Each Class: " + str(np.unique(self.total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                    print("Number of Testing Labels in Each Class: " + str(np.unique(self.total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
        else:
            raise ValueError('split_dataset(): task type error') 
     

    def build_transfer_learning_dataset(self): #this creates test dataset for transfer learning
        scene_graph_dataset  = SceneGraphDataset()
        scene_graph_dataset.dataset_save_path = self.config.location_data["transfer_path"]
        self.scene_graph_dataset = scene_graph_dataset.load()

        self.transfer_data= []
        sorted_seq = sorted(self.scene_graph_dataset.labels)
        if self.config.training_config["scenegraph_dataset_type"] == "carla":
            for ind, seq in enumerate(sorted_seq): #for each seq in labels
                data_to_append = {"sequence":self.scene_graph_dataset.process_carla_graph_sequences(self.scene_graph_dataset.scene_graphs[seq], self.feature_list, folder_name = self.scene_graph_dataset.folder_names[ind] ), "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                self.transfer_data.append(data_to_append)
                    
        elif self.config.training_config["scenegraph_dataset_type"] == "real":
            for ind, seq in enumerate(sorted_seq): 
                data_to_append = {"sequence": self.scene_graph_dataset.scene_graphs[seq], "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                self.transfer_data.append(data_to_append)
        else:
            raise ValueError('dataset_type unrecognized')
                
        self.total_transfer_data_labels = np.concatenate([np.full(len(data['sequence']), data['label']) for data in self.transfer_data])
        self.transfer_data_labels = [data['label'] for data in self.transfer_data]
        print('Running transfer learning on dataset: ', self.config.location_data["transfer_path"])
        print('total labels: ', len(self.transfer_data_labels))
        self.class_weights = torch.tensor([1.0, 1.0])


    def evaluate_transfer_learning(self, current_epoch=None):
        metrics = {}
        outputs_test, \
        labels_test, \
        acc_loss_test, \
        val_avg_prediction_frame, \
        val_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        test_inference_time, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr, _ = self.inference(self.transfer_data, self.transfer_data_labels)

        metrics['test'] = get_metrics(outputs_test, labels_test)
        metrics['test']['loss'] = acc_loss_test
        metrics['test']['avg_prediction_frame'] = val_avg_prediction_frame
        metrics['test']['avg_seq_len'] = val_avg_seq_len
        metrics['test']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['test']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['test']['seq_tpr'] = seq_tpr
        metrics['test']['seq_tnr'] = seq_tnr
        metrics['test']['seq_fpr'] = seq_fpr
        metrics['test']['seq_fnr'] = seq_fnr
        metrics['avg_inf_time'] = (test_inference_time) / (len(labels_test))


        self.update_sg_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
        metrics['best_avg_pred_frame'] = self.best_avg_pred_frame
        
        if self.config.training_config["n_fold"] <= 1 and self.log:
            log_wandb_transfer_learning(metrics)
        
        return outputs_test, labels_test, metrics


    def build_scenegraph_dataset(self): #this creates test and training datasets
        '''
        Dataset format 
            scenegraphs_sequence: dict_keys(['sequence', 'label', 'folder_name'])
                'sequence': scenegraph metadata
                'label': classification output [0 -> non_risky (negative), 1 -> risky (positive)]
                'folder_name': foldername storing sequence data
    
        Dataset modes
            no downsample
                all sequences used for the train and test set regardless of class distribution
            downsample  
                equal amount of positive and negative sequences used for the train and test set
            transfer 
                replaces original test set with another dataset 
        '''
        #Load sg dataset obj
        scene_graph_dataset  = SceneGraphDataset()
        scene_graph_dataset.dataset_save_path = self.config.location_data["input_path"]
        self.scene_graph_dataset = scene_graph_dataset.load()     
    
        class_0 = []
        class_1 = []
        sorted_seq = sorted(self.scene_graph_dataset.labels)
        
        
        if self.config.training_config["scenegraph_dataset_type"] == "carla":
            for ind, seq in enumerate(sorted_seq): #for each seq in labels
                data_to_append = {"sequence": self.scene_graph_dataset.scene_graphs[seq], "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                if self.scene_graph_dataset.labels[seq] == 0:
                    class_0.append(data_to_append) 
                elif self.scene_graph_dataset.labels[seq] == 1:
                    class_1.append(data_to_append)
                    
        elif self.config.training_config["scenegraph_dataset_type"] == "real":
            if self.config.training_config["scenegraph_data_format"] == "carla":
                for ind, seq in enumerate(sorted_seq): #for each seq in labels
                    ego_idx = sorted(self.scene_graph_dataset.scene_graphs[seq].keys())[0]
                    data_to_append = {"sequence": self.scene_graph_dataset.scene_graphs[seq], "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                    if self.scene_graph_dataset.labels[seq] == 0:
                        class_0.append(data_to_append) 
                    elif self.scene_graph_dataset.labels[seq] == 1:
                        class_1.append(data_to_append)
            
            elif self.config.training_config["scenegraph_data_format"] == "honda":
                for ind, seq in enumerate(sorted_seq): 
                    data_to_append = {"sequence": self.scene_graph_dataset.scene_graphs[seq], "label":self.scene_graph_dataset.labels[seq], "folder_name": self.scene_graph_dataset.folder_names[ind]}
                    if self.scene_graph_dataset.labels[seq] == 0:
                        class_0.append(data_to_append)  
                    elif self.scene_graph_dataset.labels[seq] == 1:
                        class_1.append(data_to_append)
            else: 
                raise ValueError('scenegraph_data_format not recognized') 

        else:
            raise ValueError('scenegraph_dataset_type not recognized') 
        
        y_0 = [0]*len(class_0)  
        y_1 = [1]*len(class_1)
        min_number = min(len(class_0), len(class_1))
        downsample = self.config.training_config["downsample"]
        
        if downsample:
            modified_class_0, modified_y_0 = resample(class_0, y_0, n_samples=min_number)
        else:
            modified_class_0, modified_y_0 = class_0, y_0
        train, test, train_y, test_y = train_test_split(modified_class_0+class_1, modified_y_0+y_1, test_size=self.config.training_config["split_ratio"], shuffle=True, stratify=modified_y_0+y_1, random_state=self.config.seed)
        if self.config.location_data["transfer_path"] != None:
            self.build_transfer_learning_dataset()
        return train, test


    '''preprocess nodes for the self supervision task'''
    def preprocess_nodes(self, nodes):
        actor_names = self.config.relation_extraction_settings["ACTOR_NAMES"]
        node_feature_list = []
        for node_id in nodes.keys():
            node_features = []
            if nodes[node_id]['label'] in ['Root Road']:
                continue
            if nodes[node_id]['label'] == 'ego car':
                nodes[node_id]['attr'].update({'left': 640-100,
                                                'top': 620,
                                                'right': 640+100,
                                                'bottom': 720})
                nodes[node_id]['label'] = "ego car_" 
            if nodes[node_id]['label'] == 'Left Lane':
                nodes[node_id]['attr'].update({'left': 0,
                                                'top': 0,
                                                'right': 0,
                                                'bottom': 0})
                nodes[node_id]['label'] = "lane_l"
            if nodes[node_id]['label'] == 'Right Lane':
                nodes[node_id]['attr'].update({'left': 1280,
                                                'top': 0,
                                                'right': 1280,
                                                'bottom': 0})
                nodes[node_id]['label'] = "lane_r" 
            if nodes[node_id]['label'] == 'Middle Lane':
                nodes[node_id]['attr'].update({'left': 640,
                                                'top': 0,
                                                'right': 640,
                                                'bottom': 0})
                nodes[node_id]['label'] = "lane_m"
            node_features = [0.0] * 15 #new format = [Left, Top, Right, Bottom, W, H, T] 
            node_features[0] = nodes[node_id]['attr']['left'] / 1280
            node_features[1] = nodes[node_id]['attr']['top'] / 720
            node_features[2] = nodes[node_id]['attr']['right'] / 1280
            node_features[3] = nodes[node_id]['attr']['bottom'] / 720
            node_features[4] = node_features[2] - node_features[0] / 1280
            node_features[5] = node_features[3] - node_features[1] / 720
            node_features[6 + actor_names.index(nodes[node_id]['label'].split('_')[0])] = 1
            node_features = torch.FloatTensor(node_features)
            node_feature_list.append(node_features)
        return node_feature_list


    def train(self):
        if (self.config.training_config['task_type'] in ['sequence_classification','graph_classification','collision_prediction']):
            tqdm_bar = tqdm(range(self.config.training_config['epochs']))
    
            for epoch_idx in tqdm_bar: # iterate through epoch   
                acc_loss_train = 0
                self.sequence_loader = DataListLoader(self.training_data, batch_size=self.config.training_config["batch_size"])
                for data_list in self.sequence_loader: # iterate through batches of the dataset
                    self.model.train()
                    self.optimizer.zero_grad()
                    labels = torch.empty(0).long().to(self.device)
                    outputs = torch.empty(0,2).to(self.device)
                    for sequence in data_list: # iterate through scene-graph sequences in the batch
                        data, label = sequence['sequence'], sequence['label'] #data= list of SceneGraph
                        new_sequence = []
                        
                        data = [data[frame] for frame in sorted(data.keys())] if type(data) == dict else data
                        for d in data: #this loop extracts a list of nodes and their attributes for each frame.
                            g = d.g
                            nodes = dict(g.nodes(data=True))
                            node_feature_list = self.preprocess_nodes(nodes)
                            new_sequence.append(torch.stack(node_feature_list).to(self.device))
                        # new_sequence : B x N x 15
                        result = self.model.forward(new_sequence)
                        output = result['output']

                        if self.config.training_config['task_type'] == 'sequence_classification': # seq vs graph based learning
                            labels  = torch.cat([labels, torch.tensor(label).long().unsqueeze(0).to(self.device)], dim=0)
                            new_sequence = torch.cat(new_sequence, dim=0)
                        elif self.config.training_config['task_type'] in ['collision_prediction']:
                            label = torch.tensor(np.full(output.shape[0], label)).long().to(self.device) #fill label to length of the sequence. shape (len_input_sequence, 1)
                            labels  = torch.cat([labels, label], dim=0)
                        else:
                            raise ValueError('task_type is unimplemented')
                        outputs = torch.cat([outputs, output.view(-1, 2)], dim=0) #in this case the output is of shape (len_input_sequence, 2)
                    loss_train = self.loss_func(outputs, labels)
                    loss_train.backward()
                    acc_loss_train += loss_train.detach().cpu().item() * len(data_list)
                    self.optimizer.step()
                    del loss_train
    
                acc_loss_train /= len(self.training_data)
                tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
    
                if epoch_idx % self.config.training_config["test_step"] == 0:
                    self.evaluate(epoch_idx)
                    
        else:
            raise ValueError('train(): task type error') 
    
                
    def cross_valid(self):
    
           # KFold cross validation with similar class distribution in each fold
           skf = StratifiedKFold(n_splits=self.config.training_config["n_fold"])
           X = np.array(self.training_data + self.testing_data)
           y = np.array(self.training_labels + self.testing_labels)
    
           # self.results stores average metrics for the the n_folds
           self.results = {}
           self.fold = 1
    
           # Split training and testing data based on n_splits (Folds)
           for train_index, test_index in skf.split(X, y):
               X_train, X_test = X[train_index], X[test_index]
               y_train, y_test = y[train_index], y[test_index]
    
               self.training_data = X_train
               self.testing_data  = X_test
               self.training_labels = y_train
               self.testing_labels  = y_test
    
               if self.config.training_config['task_type'] == 'sequence_classification':
                   print('\nFold {}'.format(self.fold))
                   print("Number of Sequences Included: ", len(self.training_data))
                   print("Num Labels in Each Class: " + str(np.unique(self.training_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
      
               elif self.config.training_config['task_type'] == 'collision_prediction':
                   print('\nFold {}'.format(self.fold))
                   print("Number of Training Sequences Included: ", len(self.training_data))
                   print("Number of Testing Sequences Included: ",  len(self.testing_data))
                   print("Number of Training Labels in Each Class: " + str(np.unique(self.total_train_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
                   print("Number of Testing Labels in Each Class: " + str(np.unique(self.total_test_labels, return_counts=True)[1]) + ", Class Weights: " + str(self.class_weights))
               
               self.best_val_loss = 99999
               self.train()
               outputs_train, labels_train, outputs_test, labels_test, metrics = self.evaluate(self.fold)
               self.update_sg_cross_valid_metrics(outputs_train, labels_train, outputs_test, labels_test, metrics)
    
               if self.fold != self.config.training_config["n_fold"]:            
                   del self.model
                   del self.optimizer
                   self.build_model()
                   
               self.fold += 1            
           del self.results
                   
                   
    def inference(self, testing_data, testing_labels):
        labels = torch.LongTensor().to(self.device)
        outputs = torch.FloatTensor().to(self.device)
        acc_loss_test = 0
        attns_weights = []
        node_attns = []
        folder_names = []
        sum_prediction_frame = 0
        sum_seq_len = 0
        num_risky_sequences = 0
        num_safe_sequences = 0
        sum_predicted_risky_indices = 0 #sum is calculated as (value * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
        sum_predicted_safe_indices = 0  #sum is calculated as ((1-value) * (index+1))/sum(range(seq_len)) for each value and index in the sequence.
        inference_time = 0 #TODO: remove this metric
        prof_result = ""
        correct_risky_seq = 0
        correct_safe_seq = 0
        incorrect_risky_seq = 0
        incorrect_safe_seq = 0
        all_graph_lists = []
        if 'log_graphs' in self.config.model_config.keys() and self.config.model_config['log_graphs']:
            os.makedirs(self.config.model_config['ss_graph_path'] + 'correct', exist_ok=True)
            os.makedirs(self.config.model_config['ss_graph_path'] + 'incorrect', exist_ok=True)
            os.makedirs(self.config.model_config['orig_graph_path'] + 'correct', exist_ok=True)
            os.makedirs(self.config.model_config['orig_graph_path'] + 'incorrect', exist_ok=True)

        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            self.model.eval()
            with torch.no_grad():
                for i in range(len(testing_data)): # iterate through scenegraphs
                    data, label = testing_data[i]['sequence'], testing_labels[i]
                    folder_name = testing_data[i]['folder_name']
                    folder_names.append(folder_name)
                    new_sequence = []

                    #this loop extracts a list of nodes and their attributes for each frame.
                    data = [data[frame] for frame in sorted(data.keys())] if type(data) == dict else data
                    for d in data:
                        g = d.g
                        nodes = dict(g.nodes(data=True))
                        node_feature_list = self.preprocess_nodes(nodes)
                        new_sequence.append(torch.stack(node_feature_list).to(self.device))
                    out = self.model.forward(new_sequence)
                    output, graph_list = out['output'], out['graph_list']
                    inference_time += 0
                    output = output.view(-1,2)
                    label = torch.tensor(np.full(output.shape[0], label)).long().to(self.device) #fill label to length of the sequence.
                    if 'log_graphs' in self.config.model_config.keys() and self.config.model_config['log_graphs']:
                        if label == output.max(1)[1].type_as(label):
                            torch.save(graph_list, self.config.model_config['ss_graph_path'] + 'correct/'+folder_name+".pt")
                            torch.save(data, self.config.model_config['orig_graph_path'] + 'correct/'+folder_name+".pt")
                        else:
                            torch.save(graph_list, self.config.model_config['ss_graph_path'] + 'incorrect/'+folder_name+".pt")
                            torch.save(data, self.config.model_config['orig_graph_path'] + 'incorrect/'+folder_name+".pt")
                        all_graph_lists.append(graph_list)
                    outputs = torch.cat([outputs, output], dim=0)
                    labels = torch.cat([labels, label], dim=0)

                    #log metrics for risky and non-risky clips separately.
                    if(1 in label):
                        preds = output.max(1)[1].type_as(label)
                        num_risky_sequences += 1
                        sum_seq_len += output.shape[0]
                        if (1 in preds):
                            correct_risky_seq += 1 #sequence level metrics
                            sum_prediction_frame += torch.where(preds == 1)[0][0].item() #returns the first index of a "risky" prediction in this sequence.
                            sum_predicted_risky_indices += torch.sum(torch.where(preds==1)[0]+1).item()/np.sum(range(output.shape[0]+1)) #(1*index)/seq_len added to sum.
                        else:
                            incorrect_risky_seq += 1
                            sum_prediction_frame += output.shape[0] #if no risky predictions are made, then add the full sequence length to running avg.
                    elif(0 in label):
                        preds = output.max(1)[1].type_as(label)
                        num_safe_sequences += 1
                        if (1 in preds): #sequence level metrics
                            incorrect_safe_seq += 1 
                        else:
                            correct_safe_seq += 1 

                        if (0 in preds):
                            sum_predicted_safe_indices += torch.sum(torch.where(preds==0)[0]+1).item()/np.sum(range(output.shape[0]+1)) #(1*index)/seq_len added to sum.

                    loss_test = self.loss_func(output, label)
                    acc_loss_test += loss_test.detach().cpu().item()

        graph_metrics = get_graph_metrics([graph for sublist in all_graph_lists for graph in sublist])
        print(graph_metrics)
        avg_risky_prediction_frame = sum_prediction_frame / num_risky_sequences if num_risky_sequences != 0 else 0 #avg of first indices in a sequence that a risky frame is first correctly predicted.
        avg_risky_seq_len = sum_seq_len / num_risky_sequences if num_risky_sequences != 0 else 0 #sequence length for comparison with the prediction frame metric. 
        avg_predicted_risky_indices = sum_predicted_risky_indices / num_risky_sequences if num_risky_sequences != 0 else 0
        avg_predicted_safe_indices = sum_predicted_safe_indices / num_safe_sequences if num_safe_sequences != 0 else 0
        seq_tpr = correct_risky_seq / num_risky_sequences if num_risky_sequences != 0 else 0
        seq_fpr = incorrect_safe_seq / num_safe_sequences if num_safe_sequences != 0 else 0
        seq_tnr = correct_safe_seq / num_safe_sequences if num_safe_sequences != 0 else 0
        seq_fnr = incorrect_risky_seq / num_risky_sequences if num_risky_sequences != 0 else 0
        if prof != None:
            prof_result = prof.key_averages().table(sort_by="cuda_time_total")

        return outputs, \
                labels, \
                acc_loss_test/len(testing_data), \
                attns_weights, \
                node_attns, \
                avg_risky_prediction_frame, \
                avg_risky_seq_len, \
                avg_predicted_risky_indices, \
                avg_predicted_safe_indices, \
                inference_time, \
                prof_result, \
                seq_tpr, \
                seq_fpr, \
                seq_tnr, \
                seq_fnr, folder_names

   


    def evaluate(self, current_epoch=None):
        metrics = {}
        outputs_train, \
        labels_train, \
        acc_loss_train, \
        attns_train, \
        node_attns_train, \
        train_avg_prediction_frame, \
        train_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        train_inference_time, \
        train_profiler_result, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr, _ = self.inference(self.training_data, self.training_labels)
        torch.cuda.empty_cache() #clear training data from GPU memory

        metrics['train'] = get_metrics(outputs_train, labels_train)
        metrics['train']['loss'] = acc_loss_train
        metrics['train']['avg_prediction_frame'] = train_avg_prediction_frame
        metrics['train']['avg_seq_len'] = train_avg_seq_len
        metrics['train']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['train']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['train']['seq_tpr'] = seq_tpr
        metrics['train']['seq_tnr'] = seq_tnr
        metrics['train']['seq_fpr'] = seq_fpr
        metrics['train']['seq_fnr'] = seq_fnr
        with open("graph_profile_metrics.txt", mode='w') as f:
            f.write(train_profiler_result)

        outputs_test, \
        labels_test, \
        acc_loss_test, \
        attns_test, \
        node_attns_test, \
        val_avg_prediction_frame, \
        val_avg_seq_len, \
        avg_predicted_risky_indices, \
        avg_predicted_safe_indices, \
        test_inference_time, \
        test_profiler_result, \
        seq_tpr, \
        seq_fpr, \
        seq_tnr, \
        seq_fnr, _ = self.inference(self.testing_data, self.testing_labels)
        
        metrics['test'] = get_metrics(outputs_test, labels_test)
        metrics['test']['loss'] = acc_loss_test
        metrics['test']['avg_prediction_frame'] = val_avg_prediction_frame
        metrics['test']['avg_seq_len'] = val_avg_seq_len
        metrics['test']['avg_predicted_risky_indices'] = avg_predicted_risky_indices
        metrics['test']['avg_predicted_safe_indices'] = avg_predicted_safe_indices
        metrics['test']['seq_tpr'] = seq_tpr
        metrics['test']['seq_tnr'] = seq_tnr
        metrics['test']['seq_fpr'] = seq_fpr
        metrics['test']['seq_fnr'] = seq_fnr
        metrics['avg_inf_time'] = (train_inference_time + test_inference_time) / (len(labels_train) + len(labels_test))

        print("\ntrain loss: " + str(acc_loss_train) + ", acc:", metrics['train']['acc'], metrics['train']['confusion'], "mcc:", metrics['train']['mcc'], \
              "\ntest loss: " +  str(acc_loss_test) + ", acc:",  metrics['test']['acc'],  metrics['test']['confusion'], "mcc:", metrics['test']['mcc'])
        
        self.update_sg_best_metrics(metrics, current_epoch)
        metrics['best_epoch'] = self.best_epoch
        metrics['best_val_loss'] = self.best_val_loss
        metrics['best_val_acc'] = self.best_val_acc
        metrics['best_val_auc'] = self.best_val_auc
        metrics['best_val_conf'] = self.best_val_confusion
        metrics['best_val_f1'] = self.best_val_f1
        metrics['best_val_mcc'] = self.best_val_mcc
        metrics['best_val_acc_balanced'] = self.best_val_acc_balanced
        metrics['best_avg_pred_frame'] = self.best_avg_pred_frame
        
        if self.config.training_config["n_fold"] <= 1 and self.log:
            log_wandb(metrics)

        torch.cuda.empty_cache() #clear test data from GPU memory

        return outputs_train, labels_train, outputs_test, labels_test, metrics


        #automatically save the model and metrics with the lowest validation loss
    def update_sg_best_metrics(self, metrics, current_epoch):
        if metrics['test']['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['test']['loss']
            self.best_epoch = current_epoch if current_epoch != None else self.config.training_config["epochs"]
            self.best_val_acc = metrics['test']['acc']
            self.best_val_auc = metrics['test']['auc']
            self.best_val_confusion = metrics['test']['confusion']
            self.best_val_f1 = metrics['test']['f1']
            self.best_val_mcc = metrics['test']['mcc']
            self.best_val_acc_balanced = metrics['test']['balanced_acc']
            self.best_avg_pred_frame = metrics['test']['avg_prediction_frame']
            self.save_model()

    
    # Averages metrics after the end of each cross validation fold
    def update_sg_cross_valid_metrics(self, outputs_train, labels_train, outputs_test, labels_test, metrics):
        if self.fold == 1:
            self.results['outputs_train'] = outputs_train
            self.results['labels_train'] = labels_train
            self.results['train'] = metrics['train']
            self.results['train']['loss'] = metrics['train']['loss']
            self.results['train']['avg_prediction_frame'] = metrics['train']['avg_prediction_frame'] 
            self.results['train']['avg_seq_len']  = metrics['train']['avg_seq_len'] 
            self.results['train']['avg_predicted_risky_indices'] = metrics['train']['avg_predicted_risky_indices'] 
            self.results['train']['avg_predicted_safe_indices'] = metrics['train']['avg_predicted_safe_indices']

            self.results['outputs_test'] = outputs_test
            self.results['labels_test'] = labels_test
            self.results['test'] = metrics['test']
            self.results['test']['loss'] = metrics['test']['loss'] 
            self.results['test']['avg_prediction_frame'] = metrics['test']['avg_prediction_frame'] 
            self.results['test']['avg_seq_len'] = metrics['test']['avg_seq_len'] 
            self.results['test']['avg_predicted_risky_indices'] = metrics['test']['avg_predicted_risky_indices'] 
            self.results['test']['avg_predicted_safe_indices'] = metrics['test']['avg_predicted_safe_indices']
            self.results['avg_inf_time'] = metrics['avg_inf_time']

            self.results['best_epoch']    = metrics['best_epoch']
            self.results['best_val_loss'] = metrics['best_val_loss']
            self.results['best_val_acc']  = metrics['best_val_acc']
            self.results['best_val_auc']  = metrics['best_val_auc']
            self.results['best_val_conf'] = metrics['best_val_conf']
            self.results['best_val_f1']   = metrics['best_val_f1']
            self.results['best_val_mcc']  = metrics['best_val_mcc']
            self.results['best_val_acc_balanced'] = metrics['best_val_acc_balanced']
            self.results['best_avg_pred_frame'] = metrics['best_avg_pred_frame']
        else:
            self.results['outputs_train'] = torch.cat((self.results['outputs_train'], outputs_train), dim=0)
            self.results['labels_train']  = torch.cat((self.results['labels_train'], labels_train), dim=0)
            self.results['train']['loss'] = np.append(self.results['train']['loss'], metrics['train']['loss'])
            self.results['train']['avg_prediction_frame'] = np.append(self.results['train']['avg_prediction_frame'], 
                                                                            metrics['train']['avg_prediction_frame'])
            self.results['train']['avg_seq_len']  = np.append(self.results['train']['avg_seq_len'], metrics['train']['avg_seq_len'])
            self.results['train']['avg_predicted_risky_indices'] = np.append(self.results['train']['avg_predicted_risky_indices'], 
                                                                                    metrics['train']['avg_predicted_risky_indices'])
            self.results['train']['avg_predicted_safe_indices'] = np.append(self.results['train']['avg_predicted_safe_indices'], 
                                                                                    metrics['train']['avg_predicted_safe_indices'])
            
            self.results['outputs_test'] = torch.cat((self.results['outputs_test'], outputs_test), dim=0)
            self.results['labels_test']  = torch.cat((self.results['labels_test'], labels_test), dim=0)
            self.results['test']['loss'] = np.append(self.results['test']['loss'], metrics['test']['loss'])
            self.results['test']['avg_prediction_frame'] = np.append(self.results['test']['avg_prediction_frame'], 
                                                                        metrics['test']['avg_prediction_frame'])
            self.results['test']['avg_seq_len'] = np.append(self.results['test']['avg_seq_len'], metrics['test']['avg_seq_len'])
            self.results['test']['avg_predicted_risky_indices'] = np.append(self.results['test']['avg_predicted_risky_indices'], 
                                                                                    metrics['test']['avg_predicted_risky_indices'])
            self.results['test']['avg_predicted_safe_indices'] = np.append(self.results['test']['avg_predicted_safe_indices'], 
                                                                                    metrics['test']['avg_predicted_safe_indices'])
            self.results['avg_inf_time'] = np.append(self.results['avg_inf_time'], metrics['avg_inf_time'])

            self.results['best_epoch']    = np.append(self.results['best_epoch'], metrics['best_epoch'])
            self.results['best_val_loss'] = np.append(self.results['best_val_loss'], metrics['best_val_loss'])
            self.results['best_val_acc']  = np.append(self.results['best_val_acc'], metrics['best_val_acc'])
            self.results['best_val_auc']  = np.append(self.results['best_val_auc'], metrics['best_val_auc'])
            self.results['best_val_conf'] = np.append(self.results['best_val_conf'], metrics['best_val_conf'])
            self.results['best_val_f1']   = np.append(self.results['best_val_f1'], metrics['best_val_f1'])
            self.results['best_val_mcc']  = np.append(self.results['best_val_mcc'], metrics['best_val_mcc'])
            self.results['best_val_acc_balanced'] = np.append(self.results['best_val_acc_balanced'], metrics['best_val_acc_balanced'])
            self.results['best_avg_pred_frame'] = np.append(self.results['best_avg_pred_frame'], metrics['best_avg_pred_frame'])
            
        # Log final averaged results
        if self.fold == self.config.training_config["n_fold"]:
            final_results = {}
            final_results['train'] = get_metrics(self.results['outputs_train'], self.results['labels_train'])
            final_results['train']['loss'] = np.average(self.results['train']['loss'])
            final_results['train']['avg_prediction_frame'] = np.average(self.results['train']['avg_prediction_frame'])
            final_results['train']['avg_seq_len'] = np.average(self.results['train']['avg_seq_len'])
            final_results['train']['avg_predicted_risky_indices'] = np.average(self.results['train']['avg_predicted_risky_indices'])
            final_results['train']['avg_predicted_safe_indices'] = np.average(self.results['train']['avg_predicted_safe_indices'])
            
            final_results['test'] = get_metrics(self.results['outputs_test'], self.results['labels_test'])
            final_results['test']['loss'] = np.average(self.results['test']['loss'])
            final_results['test']['avg_prediction_frame'] = np.average(self.results['test']['avg_prediction_frame'])
            final_results['test']['avg_seq_len'] = np.average(self.results['test']['avg_seq_len'])
            final_results['test']['avg_predicted_risky_indices'] = np.average(self.results['test']['avg_predicted_risky_indices'])
            final_results['test']['avg_predicted_safe_indices'] = np.average(self.results['test']['avg_predicted_safe_indices'])
            final_results['avg_inf_time'] = np.average(self.results['avg_inf_time'])

            # Best results
            final_results['best_epoch']    = np.average(self.results['best_epoch'])
            final_results['best_val_loss'] = np.average(self.results['best_val_loss'])
            final_results['best_val_acc']  = np.average(self.results['best_val_acc'])
            final_results['best_val_auc']  = np.average(self.results['best_val_auc'])
            final_results['best_val_conf'] = self.results['best_val_conf']
            final_results['best_val_f1']   = np.average(self.results['best_val_f1'])
            final_results['best_val_mcc']  = np.average(self.results['best_val_mcc'])
            final_results['best_val_acc_balanced'] = np.average(self.results['best_val_acc_balanced'])
            final_results['best_avg_pred_frame'] = np.average(self.results['best_avg_pred_frame'])

            print('\nFinal Averaged Results')
            print("\naverage train loss: " + str(final_results['train']['loss']) + ", average acc:", final_results['train']['acc'], final_results['train']['confusion'], final_results['train']['auc'], \
                "\naverage test loss: " +  str(final_results['test']['loss']) + ", average acc:", final_results['test']['acc'],  final_results['test']['confusion'], final_results['test']['auc'])
            if self.log:
                log_wandb(final_results)
            
            return self.results['outputs_train'], self.results['labels_train'], self.results['outputs_test'], self.results['labels_test'], final_results


