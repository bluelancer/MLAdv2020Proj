import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from graphsage.datafetcher import DataFetcher
from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

# num_sample_enc1, num_sample_enc2, num_class
sample_config = {'cora':(5,5,7), 'pubmed':(10,25,3)}
class SupervisedGraphSageLinkPred(nn.Module):

    def __init__(self, enc):
        super(SupervisedGraphSageLinkPred, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(2, enc.embed_dim * 2))
        init.xavier_uniform_(self.weight)

    def forward(self, links):
        nodes1 = []
        nodes2 = []
        for link in links:
            nodes1.append(link[0])
            nodes2.append(link[1])
        embeds1 = self.enc(nodes1)
        embeds2 = self.enc(nodes2)
        embeds_combined = torch.cat((embeds1, embeds2), 0)
        scores = self.weight.mm(embeds_combined)
        return scores.t()

    def loss(self, links, labels):
        scores = self.forward(links)
        return self.xent(scores, labels)

class SupervisedGraphSageNodeClassification(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSageNodeClassification, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def node_classification(dataset, enc2, num_nodes, labels):
    graphsage = SupervisedGraphSageNodeClassification(sample_config[dataset][2], enc2)
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.data)

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def link_prediction(enc2, num_nodes, link_list):
    graphsage = SupervisedGraphSageLinkPred(enc2)
    #    graphsage.cuda()
    # link preprocessing
    num_links = len(link_list)
    for i in range(num_links):
        paper2_rand = np.random.randint(num_nodes)
        paper1_rand = np.random.randint(num_nodes)
        if paper1_rand != paper2_rand and (paper1_rand, paper2_rand, 1) not in link_list:
            link_list.append((paper1_rand, paper2_rand, 0))
    num_links = len(link_list)

    rand_indices = np.random.permutation(num_links)
    test_link = rand_indices[:int(num_links * 0.25)]
    val_link = rand_indices[int(num_links * 0.25):int(num_links * 0.5)]
    train_link = list(rand_indices[:int(num_links * 0.5)])

    val_input, val_label, test_input, test_label = [], [], [], []
    for link_index in val_link:
        val_input.append((link_list[link_index][0], link_list[link_index][1]))
        val_label.append(link_list[link_index][2])
    for link_index in test_link:
        test_input.append((link_list[link_index][0], link_list[link_index][1]))
        test_label.append(link_list[link_index][2])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    BATCH_SIZE = 256
    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for batch in range(num_links // BATCH_SIZE):
            batch_links = train_link[:BATCH_SIZE]
            batch_input = []
            batch_output = []
            for link_index in batch_links:
                batch_input.append((link_list[link_index][0], link_list[link_index][1]))
                batch_output.append(link_list[link_index][2])
            random.shuffle(train_link)
            optimizer.zero_grad()
            loss = graphsage.loss(batch_input, Variable(torch.LongTensor(batch_output)))
            loss.backward()
            optimizer.step()
            print("Batch:", batch, "LOSS:", loss.data.detach().numpy())
        end_time = time.time()
        times.append(end_time - start_time)
        print("EPOCH:", epoch, "LOSS:", loss.data.detach().numpy())

    val_output = graphsage.forward(val_input)
    print("Validation F1:", f1_score(val_label, val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def run(dataset, mode):
    np.random.seed(1)
    random.seed(1)
    datafetcher = DataFetcher()
    feat_data, labels, adj_lists, link_list = datafetcher.load(dataset)
    num_nodes = feat_data.shape[0]
    feature_size = feat_data.shape[1]
    features = nn.Embedding(num_nodes, feature_size)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feature_size, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = sample_config[dataset][0]
    enc2.num_samples = sample_config[dataset][1]
    if mode == 'link_pred':
        link_prediction(enc2, num_nodes, link_list)
    elif mode == 'node_clas':
        node_classification(dataset, enc2, num_nodes, labels)
    else:
        print('mode not support')
        return

if __name__ == "__main__":
    run('cora', 'node_clas')
    # run_pubmed()
