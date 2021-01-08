from collections import defaultdict
from scipy.io import mmread

import numpy as np


class DataFetcher:
    def __init__(self):
        return

    def load(self, dataset):
        if dataset == 'cora':
            return self.load_cora()
        elif dataset == 'pubmed':
            return self.load_pubmed()

    def load_cora(self):
        num_nodes = 2708
        num_feats = 1433
        feat_data = np.zeros((num_nodes, num_feats))
        labels = np.empty((num_nodes, 1), dtype=np.int64)
        node_map = {}
        label_map = {}
        with open("../dataset_UNRL/citation/cora/cora.content") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                for j in range(len(info) - 2):
                    feat_data[i, j] = float(info[j + 1])
                # feat_data[i,:] = map(float(), info[1:-1])
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]

        adj_lists = defaultdict(set)
        link_list = []
        with open("../dataset_UNRL/citation/cora/cora.cites") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
                link_list.append((paper1, paper2, 1))
        return feat_data, labels, adj_lists, link_list

    def load_pubmed(self):
        # hardcoded for simplicity...
        num_nodes = 19717
        num_feats = 500
        feat_data = np.zeros((num_nodes, num_feats))
        labels = np.empty((num_nodes, 1), dtype=np.int64)
        node_map = {}
        with open("../dataset_UNRL/citation/pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
            fp.readline()
            feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
            for i, line in enumerate(fp):
                info = line.split("\t")
                node_map[info[0]] = i
                labels[i] = int(info[1].split("=")[1]) - 1
                for word_info in info[2:-1]:
                    word_info = word_info.split("=")
                    feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
        adj_lists = defaultdict(set)
        with open("../dataset_UNRL/citation/pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
            fp.readline()
            fp.readline()
            link_list = []
            for line in fp:
                info = line.strip().split("\t")
                paper1 = node_map[info[1].split(":")[1]]
                paper2 = node_map[info[-1].split(":")[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
                link_list.append((paper1, paper2, 1))
        return feat_data, labels, adj_lists, link_list

    def load_blogcata(self):
        filename = '../dataset_UNRL/soc/soc-BlogCatalog/soc-BlogCatalog.mtx'
        link_matrix = mmread(filename)
        return link_matrix


if __name__ == '__main__':
    datafetcher = DataFetcher()
    datafetcher.load_blogcata()
