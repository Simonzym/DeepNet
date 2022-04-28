import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import EdgeWeightNorm, GraphConv, SAGEConv, GINConv, GATConv, DotGatConv

num_cv = 0
cv_run = 0
class ImageGraph(DGLDataset):
    def __init__(self):
        super().__init__(name = 'image')
        
    def process(self):
        folder = ''.join(['Code/Info/graphDIF12/'])
        edges = pd.read_csv(''.join([folder, '/edges.csv']), converters = 
                      {'graph_id': lambda x: str(x)}, index_col = 0)

        graphs = pd.read_csv(''.join([folder, '/graphs.csv']), converters = 
                      {'graph_id': lambda x: str(x)}, index_col = 0)

        nodes = pd.read_csv(''.join([folder, 'cv', str(num_cv), '/run', str(cv_run), '/', set_type, '_nodes.csv']), converters = 
                      {'graph_id': lambda x: str(x)})
        
        self.graphs = []
        self.labels = []
        
        #create a graph for each graph ID from the edges table
        #first process 
        label_dict = {}
        num_nodes_dict = {}
        
        for _, row in graphs.iterrows():
            label_dict[row['graph_id']] = row['ADAS']
            num_nodes_dict[row['graph_id']] = row['num_nodes']
            
        #group edges/nodes by graph ID
        edges_group = edges.groupby('graph_id')
        nodes_group = nodes.groupby('graph_id')
        
        #for each graph id
        for graph_id in nodes_group.groups:
            #find the edges, nodes attributes, num_nodes and label
            edges_of_id = edges_group.get_group(graph_id)
            nodes_of_id = nodes_group.get_group(graph_id)
            
            src = edges_of_id['src'].to_numpy() - 1
            dst = edges_of_id['dst'].to_numpy() - 1
            weights = torch.from_numpy(edges_of_id['weight'].to_numpy())
            node_feature = nodes_of_id.loc[:, nodes.columns != 'graph_id'].to_numpy()
            node_feature = torch.from_numpy(node_feature)
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            
            #create a graph and add it to the list of graphs and labels
            g = dgl.graph((src, dst), num_nodes = num_nodes)
            g.ndata['f1'] = node_feature
            g = dgl.add_self_loop(g)
            add_weights = torch.ones(num_nodes)
            edge_weights = torch.cat((weights, add_weights))
            g.edata['w'] = edge_weights
            nodes_index = torch.nonzero(g.edges()[1] == (num_nodes-1))
            nodes_weights = edge_weights[nodes_index]
            g.ndata['w'] = nodes_weights.float()
            lw = np.concatenate((np.zeros(num_nodes-1), np.ones(1))).reshape((num_nodes, 1))
            g.ndata['lw'] = torch.from_numpy(lw).float()
            
            self.graphs.append(g)
            self.labels.append([label])
        
        #convert the label list to tensor for saving
        self.labels = torch.LongTensor(self.labels)

    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    
    def __len__(self):
        return len(self.graphs)
    


class GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, p, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats, norm='none', weight=True, bias = True)
        self.conv2 = GraphConv(h1_feats, h2_feats, norm='none', weight=True, bias = True)
        #self.conv3 = GraphConv(h2_feats, h3_feats, norm='none', weight=True, bias = True)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        #self.dropout3 = nn.Dropout(p)
        self.dense1 = nn.Linear(h2_feats, h3_feats)
        self.classify = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat, edge_weights):  
        norm_weights = EdgeWeightNorm(norm = 'right')
        norm_edge_weights = norm_weights(g, edge_weights)
        h = self.conv1(g, in_feat, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.conv2(g, h, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout2(h)
        # h = self.conv3(g, h, edge_weight = norm_edge_weights)
        # h = F.relu(h)
        # h = self.dropout3(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h', 'lw')
        out1 = self.dense1(hg)
        out1 = F.relu(out1)
        out1 = self.dropout3(out1)
        out2 = self.classify(out1)
        return  F.relu(out2)
    
class SGCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h3_feats, p, num_classes):
        super(SGCN, self).__init__()
        self.conv1 = SAGEConv(in_feats, h1_feats, bias = True, feat_drop = p, aggregator_type = 'gcn')
        self.conv2 = SAGEConv(h1_feats, h3_feats, bias = True, feat_drop = p, aggregator_type = 'gcn')
        #self.conv3 = GraphConv(h2_feats, h3_feats, norm='none', weight=True, bias = True)
        #self.dropout3 = nn.Dropout(p)
        self.classify = nn.Linear(h3_feats, num_classes)

    def forward(self, g, in_feat, edge_weights):  
        norm_weights = EdgeWeightNorm(norm = 'right')
        norm_edge_weights = norm_weights(g, edge_weights)
        h = self.conv1(g, in_feat, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight = norm_edge_weights)
        h = F.relu(h)
        # h = self.conv3(g, h, edge_weight = norm_edge_weights)
        # h = F.relu(h)
        # h = self.dropout3(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
    
class GIN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, h3_feats, p, num_classes):
        super(GIN, self).__init__()
        self.fun1 = nn.Linear(in_feats, h1_feats)
        self.fun2 = nn.Linear(h1_feats, h2_feats)
        self.conv1 = GINConv(self.fun1, 'sum')
        self.conv2 = GINConv(self.fun2, 'sum')
        
        self.BN1 = nn.BatchNorm1d(h1_feats)
        self.BN2 = nn.BatchNorm1d(h2_feats)
        self.BN3 = nn.BatchNorm1d(h3_feats)
        #self.conv3 = GraphConv(h2_feats, h3_feats, norm='none', weight=True, bias = True)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        #self.dropout3 = nn.Dropout(p)
        self.dense1 = nn.Linear(h2_feats, h3_feats)
        self.classify = nn.Linear(h3_feats, num_classes)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, g, in_feat, edge_weights):  
        norm_weights = EdgeWeightNorm(norm = 'right')
        norm_edge_weights = norm_weights(g, edge_weights)
        h = self.conv1(g, in_feat, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.conv2(g, h, edge_weight = norm_edge_weights)
        h = F.relu(h)
        h = self.dropout2(h)
        # h = self.conv3(g, h, edge_weight = norm_edge_weights)
        # h = F.relu(h)
        # h = self.dropout3(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h', 'lw')
        out1 = self.dense1(hg)
        out1 = F.relu(out1)
        out1 = self.dropout3(out1)
        return self.classify(out1)
    

#calculate accuray given model and dataset
def get_mse(dataloader, model, drange = [0, 256]):
    
    num_tests = 0
    
    sum_loss = 0
    
    for batched_graph, labels in dataloader:
        pred = model(batched_graph, 
                     batched_graph.ndata['f1'][:, drange[0]:drange[1]].float(),
                         batched_graph.edata['w'].float())
        cur_loss = F.mse_loss(pred, labels.float(), reduction = 'sum')
        sum_loss += cur_loss
        num_tests += len(labels)
        
    return sum_loss.detach().numpy()/num_tests
        
#train model, including calculating accuracy
def train_model(model, epochs = 100, lr = 0.001, drange = [0, 256], alpha = 0.1):
    
    train_mse = []
    test_mse = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = alpha)
    for epoch in range(epochs):
        for batched_graph, labels in train_dataloader:
            pred = model(batched_graph, 
                     batched_graph.ndata['f1'][:, drange[0]:drange[1]].float(),
                         batched_graph.edata['w'].float())
            loss = F.mse_loss(pred, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_one_mse = get_mse(train_dataloader, model, drange)
        test_one_mse = get_mse(test_dataloader, model, drange)
        train_mse.append(train_one_mse)
        test_mse.append(test_one_mse)

    return model, train_mse, test_mse

for cv_run in range(1,6):
    
    for num_cv in range(1,6):
        
        set_type = 'train'
        train_graph = ImageGraph()
        set_type = 'test'
        test_graph = ImageGraph()
           
        #get data loader
        train_sampler = SubsetRandomSampler(torch.arange(len(train_graph)))
        test_sampler = SubsetRandomSampler(torch.arange(len(test_graph)))
        
        train_dataloader = GraphDataLoader(train_graph, batch_size = 32, 
                                           sampler = train_sampler, drop_last = False)
        test_dataloader = GraphDataLoader(test_graph, batch_size = len(test_graph), 
                                          sampler = test_sampler, drop_last = False)
        gnn_model = GCN(640, 256, 128, 128, 0.3, 1)
        #gnn_noimg_model = GIN(3, 128, 64, 0.1, 3)
        # gnn_img_model = GIN(256, 512, 256, 128, 0, 3)
        
        _, gnn_train_mse, gnn_test_mse = train_model(gnn_model, drange = [0, 640], lr = 0.0005, alpha = 0.05)
        
        #_, noimg_train_accu, noimg_test_accu, noimg_train_loss, noimg_test_loss = train_model(gnn_noimg_model, drange = [256, 259], lr = 0.0003, alpha = 0.012)
        
        # _, img_train_accu, img_test_accu, img_train_loss, img_test_loss = train_model(gnn_img_model, drange = [0, 256], lr = 0.0003, alpha = 0.012)
        
        gnn_results = pd.DataFrame({'train_mse': gnn_train_mse,
                                    'test_mse': gnn_test_mse})
        
        # gnn_noimg_results = pd.DataFrame({'train_accu': noimg_train_accu,
        #                             'train_loss': noimg_train_loss,
        #                             'test_accu': noimg_test_accu,
        #                             'test_loss': noimg_test_loss})
        
        # gnn_img_results = pd.DataFrame({'train_accu': img_train_accu,
        #                             'train_loss': img_train_loss,
        #                             'test_accu': img_test_accu,
        #                             'test_loss': img_test_loss})
        
        hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run), '/gnn/adas.csv'])
        with open(hist_csv_file, mode='w') as f:
            gnn_results.to_csv(f)
            
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/gnn/noimg_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     gnn_noimg_results.to_csv(f)
            
        # hist_csv_file = ''.join(['Code/Info/graphDIF12/cv', str(num_cv), '/run', str(cv_run),'/gnn/img_results.csv'])
        # with open(hist_csv_file, mode='w') as f:
        #     gnn_img_results.to_csv(f)

