import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn import Linear, LSTM
from torch_geometric.nn import RGCNConv, TopKPooling, FastRGCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .rgcn_sag_pooling import RGCNSAGPooling
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


'''data-driven implementation of MRGCN (RS2G).'''

class RS2G(nn.Module):
    
    def __init__(self, config):
        super(RS2G, self).__init__()
        self.num_features = config.model_config['num_of_classes']
        self.num_relations = config.model_config['num_relations']
        self.num_classes  = config.model_config['nclass']
        self.num_layers = config.model_config['num_layers'] #defines number of RGCN conv layers.
        self.hidden_dim = config.model_config['hidden_dim']
        self.layer_spec = None if config.model_config['layer_spec'] == None else list(map(int, config.model_config['layer_spec'].split(',')))
        self.lstm_dim1 = config.model_config['lstm_input_dim']
        self.lstm_dim2 = config.model_config['lstm_output_dim']
        self.rgcn_func = FastRGCNConv if config.model_config['conv_type'] == "FastRGCNConv" else RGCNConv
        self.activation = F.relu if config.model_config['activation'] == 'relu' else F.leaky_relu
        self.pooling_type = config.model_config['pooling_type']
        self.readout_type = config.model_config['readout_type']
        self.temporal_type = config.model_config['temporal_type']
        self.device = config.model_config['device']
        self.dropout = config.model_config['dropout']
        self.edge_ext_thresh = config.model_config['edge_ext_thresh']
        self.conv = []
        total_dim = 0

        if self.layer_spec == None:
            if self.num_layers > 0:
                self.conv.append(self.rgcn_func(self.num_features, self.hidden_dim, self.num_relations).to(self.device))
                total_dim += self.hidden_dim
                for i in range(1, self.num_layers):
                    self.conv.append(self.rgcn_func(self.hidden_dim, self.hidden_dim, self.num_relations).to(self.device))
                    total_dim += self.hidden_dim
            else:
                self.fc0_5 = Linear(self.num_features, self.hidden_dim)
                total_dim += self.hidden_dim
        else:
            if self.num_layers > 0:
                print("using layer specification and ignoring hidden_dim parameter.")
                print("layer_spec: " + str(self.layer_spec))
                self.conv.append(self.rgcn_func(self.num_features, self.layer_spec[0], self.num_relations).to(self.device))
                total_dim += self.layer_spec[0]
                for i in range(1, self.num_layers):
                    self.conv.append(self.rgcn_func(self.layer_spec[i-1], self.layer_spec[i], self.num_relations).to(self.device))
                    total_dim += self.layer_spec[i]

            else:
                self.fc0_5 = Linear(self.num_features, self.hidden_dim)
                total_dim += self.hidden_dim

        if self.pooling_type == "sagpool":
            self.pool1 = RGCNSAGPooling(total_dim, self.num_relations, ratio=config.model_config['pooling_ratio'], rgcn_func=config.model_config['conv_type'])
        elif self.pooling_type == "topk":
            self.pool1 = TopKPooling(total_dim, ratio=config.model_config['pooling_ratio'])

        self.fc1 = Linear(total_dim, self.lstm_dim1)
        
        if "lstm" in self.temporal_type:
            self.lstm = LSTM(self.lstm_dim1, self.lstm_dim2, batch_first=True)
            self.attn = Attention(self.lstm_dim2)
            self.lstm_decoder = LSTM(self.lstm_dim2, self.lstm_dim2, batch_first=True)
        else:
            self.fc1_5 = Linear(self.lstm_dim1, self.lstm_dim2)

        self.fc2 = Linear(self.lstm_dim2, self.num_classes)

        #~~~~~~~~~~~~Data-Driven Graph Encoders~~~~~~~~~~~~~~
        #node encoder
        if config.model_config['node_encoder_dim'] == 1:
            self.node_encoder = Linear(15, self.num_features) 
        elif config.model_config['node_encoder_dim'] == 2:
            self.node_encoder = nn.Sequential(
                        nn.Linear(15, 30),
                        nn.ReLU(),
                        nn.Linear(30, self.num_features)
                    )

        #edge encoder. takes in two node embeddings and returns multilabel edge selection.
        if config.model_config['edge_encoder_dim'] == 1:
            self.edge_encoder = Linear(2 * 15, self.num_relations)
        elif config.model_config['edge_encoder_dim'] == 2:
            self.edge_encoder = nn.Sequential(
                        nn.Linear(2 * 15, 30),
                        nn.ReLU(),
                        nn.Linear(30, self.num_relations)
                    )


    def forward(self, sequence):
        #graph extraction component
        graph_list = []
        for i in range(len(sequence)):
            graph = {}
            node_feature_list = sequence[i]
            graph['node_embeddings'] = self.activation(self.node_encoder(node_feature_list))
            graph['edge_attr'] = []
            graph['edge_index'] = []

            new_arr = torch.ones([len(node_feature_list), len(node_feature_list)]).triu(diagonal=1)
            new_arr_idx = torch.where(new_arr==1.0)
            combo_list = torch.stack(new_arr_idx).t()

            new_arr_2 = new_arr.flatten().int()
            new_arr_idx2 = torch.where(new_arr_2==1.0)
            node_combo_a = node_feature_list.unsqueeze(0).repeat((node_feature_list.size(0), 1,1))
            node_combo_b = node_feature_list.unsqueeze(1).repeat((1, node_feature_list.size(0),1))
            node_combo = torch.cat([node_combo_b, node_combo_a], dim=-1).flatten(start_dim=0, end_dim=1)
            node_combinations = node_combo[new_arr_idx2]
            edge_vectors = self.edge_encoder(node_combinations)
            edge_vectors = torch.sigmoid(edge_vectors) #sigmoid to generate multilabel conf. scores, then binarize. 
            top_edges = torch.argmax(edge_vectors, dim=1) #get highest scoring edge.
            graph['edge_index'] = torch.cat([combo_list, combo_list.flip(1)], dim=0) #make edges bidirectional
            graph['edge_attr'] = torch.cat([top_edges, top_edges], dim=0) #make edges bidirectional
            pos_idxs = edge_vectors > self.edge_ext_thresh #add all edge types that score > threshold
            pos_idxs = pos_idxs.nonzero()
            pos_edge_idx, pos_edge_attrs = combo_list[pos_idxs[:,0]], pos_idxs[:, 1]
            graph['edge_index'] = torch.cat([graph['edge_index'], pos_edge_idx, pos_edge_idx.flip(1)], dim=0)
            graph['edge_attr'] = torch.cat([graph['edge_attr'], pos_edge_attrs, pos_edge_attrs], dim=0)
            graph['edge_index'] = torch.transpose(graph['edge_index'], 0, 1) 
            graph['edge_attr'] = graph['edge_attr']
            graph_list.append(graph)
        graph_data_list = [Data(x=g['node_embeddings'], edge_index=g['edge_index'], edge_attr=g['edge_attr']) for g in graph_list]
        train_loader = DataLoader(graph_data_list, batch_size=len(graph_data_list))
        sequence = next(iter(train_loader)).to(self.device)
        x, edge_index, edge_attr, batch = sequence.x, sequence.edge_index, sequence.edge_attr, sequence.batch

        #MRGCN component. downstream task
        attn_weights = dict()
        outputs = []
        if self.num_layers > 0:
            for i in range(self.num_layers):
                x = self.activation(self.conv[i](x, edge_index, edge_attr))
                x = F.dropout(x, self.dropout, training=self.training)
                outputs.append(x)
            x = torch.cat(outputs, dim=-1)
        else:
            x = self.activation(self.fc0_5(x))

        if self.pooling_type == "sagpool":
            x, edge_index, _, attn_weights['batch'], attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        elif self.pooling_type == "topk":
            x, edge_index, _, attn_weights['batch'], attn_weights['pool_perm'], attn_weights['pool_score'] = self.pool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        else: 
            attn_weights['batch'] = batch

        if self.readout_type == "add":
            x = global_add_pool(x, attn_weights['batch'])
        elif self.readout_type == "mean":
            x = global_mean_pool(x, attn_weights['batch'])
        elif self.readout_type == "max":
            x = global_max_pool(x, attn_weights['batch'])
        else:
            pass

        x = self.activation(self.fc1(x))
    
        #temporal modeling
        if self.temporal_type == "mean":
            x = self.activation(self.fc1_5(x.mean(axis=0)))
        elif self.temporal_type == "lstm_last":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = h.flatten()
        elif self.temporal_type == "lstm_sum":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x = x_predicted.sum(dim=1).flatten()
        elif self.temporal_type == "lstm_attn":
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0))
            x, attn_weights['lstm_attn_weights'] = self.attn(h.view(1,1,-1), x_predicted)
            x, (h_decoder, c_decoder) = self.lstm_decoder(x, (h, c))
            x = x.flatten()
        elif self.temporal_type == "lstm_seq": #used for step-by-step sequence prediction. 
            x_predicted, (h, c) = self.lstm(x.unsqueeze(0)) #x_predicted is sequence of predictions for each frame, h is hidden state of last item, c is last cell state
            x = x_predicted.squeeze(0) #we return x_predicted as we want to know the output of the LSTM for each value in the sequence
        else:
            pass

        return {'output': F.log_softmax(self.fc2(x), dim=-1), 
                'graph_list': graph_list}