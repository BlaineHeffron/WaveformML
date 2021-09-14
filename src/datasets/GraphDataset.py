from typing import List
import logging

from torch_geometric.data import Dataset, Data
import torch

import os.path as osp

from torch_geometric.nn import knn_graph

from src.engineering.PSDDataModule import PSDDataModule

class DataExtra(Data):
    def __init__(self, **kwargs):
        super(DataExtra, self).__init__(**kwargs)
        self.additional_fields = []

class GraphDataset(Dataset):
    def __init__(self, dataset, file_list: List, k, use_self_loops):
        root = osp.dirname(file_list[0])
        procdir = osp.join(root, "processed")
        self.expected_file_names = [osp.join(procdir, osp.basename(f)[0:-3] + ".pt") for f in file_list]
        self.raw_dataset = dataset
        self.file_indices_to_process = [i for i in range(len(file_list))]
        self.raw_file_list = file_list
        self.log = logging.getLogger(__name__)
        self.k = k
        self.use_self_loops = use_self_loops
        super().__init__(root=root)

    def process(self):
        while len(self.file_indices_to_process) > 0:

            ind = self.file_indices_to_process.pop()
            self.log.info("creating graph data from file {0} with {1} nearest neighbors".format(self.raw_file_list[ind], self.k))
            if osp.exists(self.processed_file_names[ind]):
                self.log.info("file {0} already exists, skipping".format(self.processed_file_names[ind]))
                continue
            (c, f), target = self.raw_dataset[ind]
            additional_fields = None
            if isinstance(f, list):
                additional_fields = f[1:]
                f = f[0]
            coo = c.long()
            #edge_index = knn_graph(coo, self.k, coo[:, 2], loop=self.use_self_loops)
            data = DataExtra(x=f, pos=coo, y=target)
            data.additional_fields = additional_fields
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, self.processed_file_names[ind])
            self.log.info("created file {0}".format(self.processed_file_names[ind]))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.processed_file_names[idx])

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return self.expected_file_names
