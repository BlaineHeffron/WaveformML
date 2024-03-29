from src.engineering.PSDDataModule import PSDDataModule
from torch_geometric.data import DataLoader
from src.datasets.GraphDataset import GraphDataset
from src.utils.util import DictionaryUtility


class GraphDataModule(PSDDataModule):
    def __init__(self, config, device):
        super(GraphDataModule, self).__init__(config, device)
        if not hasattr(self.config.net_config.hparams, "self_loop"):
            self.self_loop = False
        else:
            self.self_loop = self.config.net_config.hparams.self_loop
        self.graph_train_dataset = None
        self.graph_val_dataset = None
        self.graph_test_dataset = None

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage=None):
        if not hasattr(self, "train_dataset"):
            super(GraphDataModule, self).setup(None)
        if (stage is None or stage == "fit") and self.graph_test_dataset is None:
            flist = self.train_dataset.get_file_list()
            self.graph_train_dataset = GraphDataset(self.train_dataset, flist, self.self_loop)
        elif stage is None or stage == "test":
            if self.graph_val_dataset is None:
                flist = self.val_dataset.get_file_list()
                self.graph_val_dataset = GraphDataset(self.val_dataset, flist, self.self_loop)
            if self.graph_test_dataset is None:
                flist = self.test_dataset.get_file_list()
                self.graph_test_dataset = GraphDataset(self.test_dataset, flist, self.self_loop)

    def train_dataloader(self):
        if not self.graph_train_dataset:
            self.setup("train")
        return DataLoader(self.graph_train_dataset, shuffle=True,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

    def val_dataloader(self):
        if not self.graph_val_dataset:
            self.setup("test")
        return DataLoader(self.graph_val_dataset, shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

    def test_dataloader(self):
        if not self.graph_test_dataset:
            self.setup("test")
        return DataLoader(self.graph_test_dataset, shuffle=False,
                          **DictionaryUtility.to_dict(self.config.dataset_config.dataloader_params))

